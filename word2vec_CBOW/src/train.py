import torch

def train_loop(model,  dataloader,  optimizer, loss_function, device,epochs = 10, save_model = True):
    model.train()
    losses = []
    model = model.to(device=device)  
    for e in range(epochs):
        agr_loss = 0
        for i, (context, target) in enumerate(dataloader, 1):
            context = context.to(device=device)
            target = target.to(device=device)
             
            out = model(context)
            loss = loss_function(out, target)
            
            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
            
            agr_loss += loss.item()
            if i % 20000 == 0:
                print(agr_loss/i)
        loss_epoch = agr_loss/ len(dataloader)
        losses.append(loss_epoch)
        if save_model: # used to save model after every iteration on colab
            model_name = F"CBOW_e{e+1}_l_{round(loss_epoch, 3)}.pt" 
            Path = F"/content/gdrive/My Drive/{model_name}"
            torch.save(model.state_dict(),Path)    
    return losses

def train_loopNS(model,  dataloader,  optimizer, loss_function, device,epochs = 10, save_model = True):
    model.train()
    losses = []
    model = model.to(device=device)  
    for e in range(epochs):
        agr_loss = 0
        for i, (context, target, neg_samples) in enumerate(dataloader, 1):
            context = context.to(device=device)
            target = target.to(device=device)
             
            loss = model(context, target, neg_samples)

            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
            
            agr_loss += loss.item()
            if i % 200 == 0:
                print(agr_loss/i)
        loss_epoch = agr_loss/ len(dataloader)
        losses.append(loss_epoch)
        if save_model: # used to save model after every iteration on colab
            model_name = F"CBOW_e{e+1}_l_{round(loss_epoch, 3)}.pt" 
            Path = F"/content/gdrive/My Drive/{model_name}"
            torch.save(model.state_dict(),Path)    
    return losses