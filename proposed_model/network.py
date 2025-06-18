import torch.nn as nn
from torchvision.models import resnet18
import torch
class ViTDETRLPR(nn.Module):
    def __init__(self, vocab_size=68, max_seq_len=8, hidden_dim=96, num_decoder_layers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        resnet = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

        self.encoder_proj = nn.Linear(512, hidden_dim)
        self.max_tokens = 400
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_tokens, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=2, dim_feedforward=128, dropout=dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        self.sos_token = nn.Parameter(torch.randn(1, hidden_dim))

    def forward(self, x, tgt_seq=None, teacher_forcing=False, scheduled_sampling_prob=0.0):
        B = x.size(0)

        feat_map = self.backbone(x)
        bbox_out = self.bbox_head(feat_map)

        enc = feat_map.flatten(2).permute(0, 2, 1) 
        L = enc.size(1)
        enc = self.encoder_proj(enc) + self.pos_embed[:, :L, :]
        enc = enc.permute(1, 0, 2)  

        if tgt_seq is not None and teacher_forcing and scheduled_sampling_prob > 0.0:
            
            logits = []
            input_token = self.sos_token.repeat(B, 1).unsqueeze(0)  
            generated = []

            for t in range(self.max_seq_len):
                tgt_input = torch.cat(generated + [input_token], dim=0)  
                mask = nn.Transformer().generate_square_subsequent_mask(tgt_input.size(0)).to(x.device)
                dec_out = self.decoder(tgt_input, enc, tgt_mask=mask)
                out_step = self.output_head(dec_out[-1]) 
                logits.append(out_step)
                
                sampled = torch.bernoulli(torch.full((B,), scheduled_sampling_prob, device=x.device))
                
                use_pred = sampled.bool()
                
                next_token = tgt_seq[:, t] if tgt_seq is not None else torch.zeros(B, dtype=torch.long, device=x.device)
                pred_token = out_step.argmax(dim=1)

                chosen_token = torch.where(use_pred, pred_token, next_token)
                input_token = self.embedding(chosen_token).unsqueeze(0)
                generated.append(input_token)

            logits = torch.stack(logits, dim=1)  

        elif tgt_seq is not None and teacher_forcing:
            
            tgt = self.embedding(tgt_seq).permute(1, 0, 2)  
            mask = nn.Transformer().generate_square_subsequent_mask(tgt.size(0)).to(x.device)
            dec_out = self.decoder(tgt, enc, tgt_mask=mask)
            logits = self.output_head(dec_out).permute(1, 0, 2) 

        else:
            
            logits = []
            input_token = self.sos_token.repeat(B, 1).unsqueeze(0)  
            generated = []

            for t in range(self.max_seq_len):
                tgt_input = torch.cat(generated + [input_token], dim=0)
                mask = nn.Transformer().generate_square_subsequent_mask(tgt_input.size(0)).to(x.device)
                dec_out = self.decoder(tgt_input, enc, tgt_mask=mask)
                out_step = self.output_head(dec_out[-1])  
                logits.append(out_step)

                pred_token = out_step.argmax(dim=1)
                input_token = self.embedding(pred_token).unsqueeze(0)
                generated.append(input_token)

            logits = torch.stack(logits, dim=1) 

        return bbox_out, logits


