import torch
import clip
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from TNGCLIP import TNG_CLIP
from dataset import COCOCaptionDataset
from util import  save_model, prepair_similar_data



def train_simulate_relevant_pnh(model, dataloader, optimizer, epochs, save_name):
    # instead of only have negative and hybrid, make text contains positive, negative and hybrid
    def merge_into_triplets(two_list, one_list):
        assert len(two_list) % 2 == 0
        assert len(one_list) == len(two_list) // 2
        
        merged = []
        for i in range(len(one_list)):
            pair = two_list[2 * i : 2 * i + 2]   # get two from A
            third = one_list[i]                 # get one from B
            merged.extend(pair + [third])       # make a triplet
        return merged

    model.train()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        scaler = torch.cuda.amp.GradScaler()
        for i, batch in enumerate(tqdm(dataloader)):
            images = batch['image'].to(device)
            texts = batch['positive_caption']
            positive_texts = texts
            image_ids = batch['image_id']
            for j in range(len(texts)):
                if "0.09 0.23 0.29 0.42 0.95 0.88 0.72 0.69 0.51" in texts[j]:
                    texts[j] = "None"
            image_features = model.model.encode_image(images)
            image_features = F.normalize(image_features).to("cuda")
            texts, full_negation = prepair_similar_data(texts,image_ids,image_features)
         
            for k in range(len(texts)//4):
                texts[2*k+1] = texts[len(texts)-2*k-1]
            texts = texts[:len(texts)//2]
            texts = merge_into_triplets(texts,positive_texts)
            # to collect a fixed negation dataset and compare dynamic training and fixed training
            optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast():
                outputs, loss, _ = model(image_feature=image_features, text=texts)

            print(f"Epoch {epoch}, Step {i}: loss = {loss.item():.4f}", flush=True)
            # Backward pass
            scaler.scale(loss).backward()
            #loss.backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        #with open("./fixed_nph_dataset_MSCOCO_" + str(epoch) + ".json", "w") as f:
        #    json.dump(save_dataset, f, indent=4)
        #scheduler.step()
        save_path = './trained_models/'+save_name+'_epoch' + str(epoch) + '.pth'
        save_model(model, save_path)
        torch.save(model.model.state_dict(),'./'+save_path.split(".pth")[0]+'inner.pth')
        #accuracy = compute_match_accuracy(model.model, val_dataloader, device, gt_image_feature, gt_image_name)


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name = 'ViT-B/32'
_, preprocess = clip.load(clip_model_name, device=device, jit=False)


if __name__ == "__main__":

    # config area
    model_name = 'ft' # lora or ft
    batch_size = 128
    warmup_steps = 30
    total_steps = 300
    lr = 5e-6
    # ft random random_nph, randmo_aph_cn, cos_sim_nph, cos_sim_fix
    train_type = 'cos_sim_nph'
    # CC3M MSCOCO MSCOCO_80k MSCOCO_fix
    data_type = 'MSCOCO'
    special_comment = 'bs256'
    save_name = train_type + "_" + data_type + "_" + special_comment + "_" + str(batch_size) + "_" + str(lr)
    clip_model_name = 'ViT-B/32'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load(clip_model_name, device=device, jit=False)

    gt_preprocess = preprocess
    CLIP_model = TNG_CLIP(model_name=clip_model_name,batch_size=batch_size).to(device)
    
    train_image_folder = "/path/to/train_2014/fodler" 
    train_text_file = "/path/to/captinos_train2014.json" 
    train_dataset = COCOCaptionDataset(train_image_folder, train_text_file, preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   
    optimizer = torch.optim.AdamW(CLIP_model.parameters(), lr=lr)
   
    train_simulate_relevant_pnh(CLIP_model,train_dataloader,optimizer,10, save_name)
    
    