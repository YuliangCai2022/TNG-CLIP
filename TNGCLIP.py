import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# Custom CLIP Module with Self-Defined Loss

class TNG_CLIP(nn.Module):
    def __init__(self, model_name="ViT-B/32", temperature=0.07, device='cuda', batch_size=256):
        super(TNG_CLIP, self).__init__()
        self.model, _ = clip.load(model_name, device=device, jit=False)
        for param in self.model.visual.parameters():
            param.requires_grad = False
        self.model = self.model.float()
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size

    def forward(self, text, image_feature=None, images=None):
      
        image_features = image_feature
        text_tokens = clip.tokenize(text).to(self.device)
        raw_text_features = self.model.encode_text(text_tokens)
        # Normalize features
        text_features = F.normalize(raw_text_features)
      
        # Compute cosine similarity
        logits = text_features@ image_features.T / self.temperature
        loss = self.compute_loss(logits) 
        return logits, loss, raw_text_features

    def compute_loss(self, logits):
       
           
        batch_size = logits.shape[0]//3 #self.batch_size
        labels = torch.arange(batch_size, device=self.device)
        labels_text = torch.randperm(3 * batch_size)[:batch_size].to('cuda')
        logits_align = logits[:batch_size*3,:batch_size]

        loss_img = F.cross_entropy(logits_align.T, labels_text)
        loss_txt = F.cross_entropy(logits_align, labels.repeat_interleave(3))
        
        # Final loss is the average of both losses
        loss = (loss_img + loss_txt) / 2
        return loss
    
       