import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.utils.tensorboard import SummaryWriter 

from .model import *
from .dataset import *
from .config import *
from .train import *

##########################################################################################
################################## CALLBACKS #############################################
##########################################################################################

class save_model(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.config = get_config()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # save the model at the end of every epoch
        model_filename = get_weights_file_path(self.config, f"{trainer.current_epoch:02d}")
        torch.save({"epoch": trainer.current_epoch,
                    "model_state_dict": pl_module.state_dict(),
                    "optimizer_state_dict": pl_module.optimizer.state_dict(),
                    "gloabl_step": trainer.global_step}, model_filename)


##########################################################################################
################################## LIGHTNING MODULE ######################################
##########################################################################################

class CustomTransformer(pl.LightningModule):

    def __init__(self, config, lr_value=0):
        super().__init__()
        self.config = get_config()
        _, _, self.tokenizer_src, self.tokenizer_tgt = get_ds(self.config)
        self.model = get_model(config, self.tokenizer_src.get_vocab_size(), self.tokenizer_tgt.get_vocab_size()).to(self.device)
        self.initial_epoch = 0 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], eps=1e-9)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
        self.val_predicted = []
        self.val_expected = []
        self.val_print_count = 0

        # Tensorboard 
        self.writer = SummaryWriter(config['experiment_name'])                   
        #Loss array
        self.train_losses =[] 
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return self.optimizer

    # Load model checkpoint
    def on_train_start(self):
            if self.config['preload']:
                model_filename = get_weights_file_path(self.config, self.config['preload'])
                state = torch.load(model_filename)
                self.model.load_state_dict(state['optimizer_state_dict'])
                self.initial_epoch = state['epoch'] + 1
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                print("Preloaded")

    def greedy_decode(self, model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
        sos_idx = tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = tokenizer_tgt.token_to_id("[EOS]")
        # Precompute the encoder output and reuse it for every step
        encoder_output = model.encode(source, source_mask)
        # Initialize the decoder input with the start of sentence token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break
            # build mask for target
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            # calculate output
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            # get next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
            if next_word == eos_idx:
                break
        return decoder_input.squeeze(0)

    def train_dataloader(self):
        train_loader, _, _, _ = get_ds(self.config) 
        return train_loader

    def val_dataloader(self):
        _, val_loader, _, _ = get_ds(self.config)
        return val_loader

    def training_step(self, batch, batch_idx):
            encoder_input = batch['encoder_input']
            decoder_input = batch['decoder_input']
            encoder_mask = batch['encoder_mask']
            decoder_mask = batch['decoder_mask']
            label = batch['label']
            encoder_output = self.model.encode(encoder_input, encoder_mask)
            decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = self.model.project(decoder_output)
            loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
            # Calling self.log will surface up scalars for you in TensorBoard
            self.log("train_loss", loss.item(), prog_bar=True)             
            self.train_losses.append(loss.item())         
            # Logging loss 
            self.writer.add_scalar('train,loss', loss.item(), self.trainer.global_step) 
            self.writer.flush()                 
            # Backpropagate the loss 
            loss.backward(retain_graph=True) 
            return loss

    def validation_step(self, batch, batch_idx):       
            max_len = self.config['seq_len'] 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
            num_examples=0
            with torch.no_grad():             
                encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
                encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
               # check that the batch size is 1 
                assert encoder_input.size(0) == 1, "Batch  size must be 1 for val"
                model_out = self.greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_src, self.tokenizer_tgt, max_len, device)
                source_text = batch["label"][0]
                target_text = batch["tgt_text"][0] 
                model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())
                
                self.val_expected.append(target_text) 
                self.val_predicted.append(model_out_text) 
                org_text = self.tokenizer_src.decode(batch["encoder_input"][0].detach().cpu().numpy())
                # Check if printed examples count is less than 2
                if self.val_print_count < 2:
                    # Print the source, target, and model output             
                    print(f"{f'SOURCE: ':>12}{org_text}") 
                    print(f"{f'TARGET: ':>12}{target_text}")
                    print(f"{f'PREDICTED: ':>12}{model_out_text}")
                    print("\n")
                    # Increase the counter
                    self.val_print_count += 1
    
    def on_validation_epoch_start(self):
        self.val_print_count = 0

    def on_validation_epoch_end(self):
            writer = self.writer
            if writer:
                # Evaluate the character error rate 
                # Compute the char error rate 
                metric = torchmetrics.CharErrorRate() 
                cer = metric(self.val_predicted, self.val_expected) 
                writer.add_scalar('validation cer', cer, self.trainer.global_step) 
                writer.flush() 
            
                # Compute the word error rate 
                metric = torchmetrics.WordErrorRate() 
                wer = metric(self.val_predicted, self.val_expected) 
                writer.add_scalar('validation wer', wer, self.trainer.global_step) 
                writer.flush() 
            
                # Compute the BLEU metric 
                metric = torchmetrics.BLEUScore() 
                bleu = metric(self.val_predicted, self.val_expected) 
                writer.add_scalar('validation BLEU', bleu, self.trainer.global_step) 
                writer.flush() 

