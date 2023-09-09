import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.tensorboard import SummaryWriter

from .model import *
from .dataset import *
from .config import *
from .train import *

class save_model(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.config = get_config()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # save the model at the end of every epoch
        model_filename = get_weights_file_path(self.config, f"{trainer.current_epoch:02d}")
        torch.save({
            "epoch": trainer.current_epoch,
            "model_state_dict": pl_module.state_dict(),
            "optimizer_state_dict": pl_module.optimizer.state_dict(),
            "global_step": trainer.global_step
        }, model_filename)

class CustomTransformer(pl.LightningModule):

    def __init__(self, config, lr_value=0):
        super().__init__()
        self.config = get_config()
        _, _, self.tokenizer_src, self.tokenizer_tgt = get_ds(self.config)
        self.model = get_model(config, self.tokenizer_src.get_vocab_size(), self.tokenizer_tgt.get_vocab_size()).to(self.device)
        self.initial_epoch = 0 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], eps=1e-9)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
        self.val_predicted = []
        self.val_expected = []
        self.val_print_count = 0
        self.writer = SummaryWriter(config['experiment_name'])
        self.train_losses = []

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        if self.config['preload']:
            model_filename = get_weights_file_path(self.config, self.config['preload'])
            state = torch.load(model_filename)
            self.model.load_state_dict(state['model_state_dict'])
            self.initial_epoch = state['epoch'] + 1
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            print("Preloaded")

    def greedy_decode(self, model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
        sos_idx = tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = tokenizer_tgt.token_to_id("[EOS]")
        encoder_output = model.encode(source, source_mask)
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
        while True:
            if decoder_input.size(1) == max_len:
                break
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
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
        self.log("train_loss", loss.item(), prog_bar=True)
        self.train_losses.append(loss.item())
        self.writer.add_scalar('train_loss', loss.item(), self.trainer.global_step)
        self.writer.flush()
        loss.backward(retain_graph=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"], eps=1e-9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config["max_lr"],
            epochs=self.config["num_epochs"],
            pct_start=1/10 if self.config["num_epochs"] != 1 else 0.5,
            steps_per_epoch=len(self.train_dataloader()),
            div_factor=10,
            three_phase=True,
            final_div_factor=10,
            anneal_strategy="linear"
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    # def validation_step(self, batch, batch_idx):
    #     max_len = self.config['seq_len']
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     num_examples = 0
    #     with torch.no_grad():
    #         encoder_input = batch["encoder_input"].to(device)
    #         encoder_mask = batch["encoder_mask"].to(device)
    #         assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
    #         model_out = self.greedy_decode(self.model, encoder_input, encoder_mask, self.tokenizer_src, self.tokenizer_tgt, max_len, device)
    #         source_text = batch["label"][0]
    #         target_text = batch["tgt_text"][0]
    #         model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    #         self.val_expected.append(target_text)
    #         self.val_predicted.append(model_out_text)
    #         org_text = self.tokenizer_src.decode(batch["encoder_input"][0].detach().cpu().numpy())
    #         if self.val_print_count < 2:
    #             print(f"{f'SOURCE: ':>12}{org_text}")
    #             print(f"{f'TARGET: ':>12}{target_text}")
    #             print(f"{f'PREDICTED: ':>12}{model_out_text}")
    #             print("\n")
    #             self.val_print_count += 1

    # def on_validation_epoch_start(self):
    #     self.val_print_count = 0

    # def on_validation_epoch_end(self):
    #     writer = self.writer
    #     if writer:
    #         metric = torchmetrics.CharErrorRate()
    #         cer = metric(self.val_predicted, self.val_expected)
    #         writer.add_scalar('validation cer', cer, self.trainer.global_step)
    #         writer.flush()
            
    #         metric = torchmetrics.WordErrorRate()
    #         wer = metric(self.val_predicted, self.val_expected)
    #         writer.add_scalar('validation wer', wer, self.trainer.global_step)
    #         writer.flush()
            
    #         metric = torchmetrics.BLEUScore()
    #         bleu = metric(self.val_predicted, self.val_expected)
    #         writer.add_scalar('validation BLEU', bleu, self.trainer.global_step
