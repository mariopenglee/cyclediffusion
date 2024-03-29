from multiprocessing import process
from src.data_preparation.hico_dataset import HICO, split_dataset
from models.pretrained.pix2struct import processor, model
from src.training.training_utils import collator
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

CHECKPOINT_PATH = "checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Initialize process group
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Save checkpoint
def save_checkpoint(model, epoch, optimizer, file_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, file_path)


# Load checkpoint
def load_checkpoint(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']




# Training loop and setup
def train_pix2struct(rank, world_size, model=model, processor=processor):
    setup(rank, world_size)


    full_train_dataset = HICO(split='train')
    full_train_indices = list(range(len(full_train_dataset)))
    optimizer = AdamW(model.parameters(), lr=1e-5)



    # Split indices for training and validation
    train_indices, val_indices = split_dataset(full_train_indices, val_size=0.2)

    # Create dataset instances for training and validation
    train_dataset = HICO(split='train', indices=train_indices)
    val_dataset = HICO(split='train', indices=val_indices)

    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, collate_fn=collator)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(val_dataset, batch_size=8, sampler=val_sampler, collate_fn=collator)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
   # freeze all layers except last and langauge output layer
    for name, param in model.named_parameters():
        param.requires_grad = False

    # unfreeze_layers = ['decoder.layer.11', 'decoder.final_layer_norm', 'decoder.lm_head']
    unfreeze_layers = ['decoder.lm_head']
    for name, param in model.named_parameters():
        if any(layer in name for layer in unfreeze_layers):
            param.requires_grad = True
    
    # Load checkpoint if available
    if os.path.exists(os.path.join(CHECKPOINT_PATH, "best_model.pt")):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, os.path.join(CHECKPOINT_PATH, "best_model.pt"))
        print(f"Resuming training from epoch {start_epoch}")
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    # Training loop


    # Hyperparameters
    # Initialize TensorBoard writer
    if rank == 0:
        writer = SummaryWriter('runs/pix2struct_experiment')

    EPOCHS = 5000
    patience = 10  # Number of epochs to wait for improvement before stopping
    best_loss = float('inf')  # Initialize best loss to a very high value
    patience_counter = 0  # Initialize patience counter
    
    start_epoch = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
    accumulation_steps = 4
    for epoch in range(start_epoch, EPOCHS):
        try:
            print("Epoch:", epoch)
            model.train()  # Set the model back to training mode
            total_loss = 0
            optimizer.zero_grad()
            train_progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch} Training")
            for idx, batch in train_progress_bar:
                labels = batch.pop("labels").to(device)
                flattened_patches = batch.pop("flattened_patches").to(device)
                attention_mask = batch.pop("attention_mask").to(device)

                
                outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                if (idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss.item()
                train_progress_bar.set_description(f"Epoch {epoch} Training Loss: {loss.item():.4f}")
                train_progress_bar.refresh()

            # Compute average loss for the epoch
            avg_train_loss = total_loss / len(train_dataloader)
            if rank == 0:
                writer.add_scalar('Loss/train', avg_train_loss, epoch)
            print(f"Average Training Loss: {avg_train_loss}")

            # Validation step
            model.eval()
            val_progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Epoch {epoch} Validation")

            with torch.no_grad():
                total_val_loss = 0
                for idx, batch in val_progress_bar:
                    labels = batch.pop("labels").to(device)
                    flattened_patches = batch.pop("flattened_patches").to(device)
                    attention_mask = batch.pop("attention_mask").to(device)

                    outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    val_progress_bar.set_description(f"Epoch {epoch} Validation Loss: {loss.item():.4f}")
                    val_progress_bar.refresh()

                avg_val_loss = total_val_loss / len(val_dataloader)
                print(f"Validation Loss: {avg_val_loss}")

                # Early stopping check
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0  # Reset patience counter
                    # Save the model if it's the best so far
                    if rank == 0:
                        save_checkpoint(model, epoch, optimizer, os.path.join(CHECKPOINT_PATH, "best_model.pt"))
                else:
                    patience_counter += 1
            if rank == 0:
                writer.add_scalar("Loss/val", avg_val_loss, epoch)
            scheduler.step(avg_val_loss)

            if patience_counter >= patience:
                print("Early stopping triggered")
                break  # Exit the training loop

            
        except Exception as e:
            print(f"Error during training: {e}")
            # Optionally log the error details somewhere
            continue  # Or break, based on your preference
    if rank == 0:
        writer.close()
    cleanup()

# cleanup
def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2  # Number of GPUs
    mp.spawn(train_pix2struct,
            args=(world_size,),
            nprocs=world_size,
            join=True)
    
