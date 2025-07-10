import time
import torch.nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate_accuracy(data_iter, net, device=device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            #print(X.type(),X.size())
            y=y.view(1,-1)[0]
            y = y.to(dtype=torch.long) 
            if isinstance(net, torch.nn.Module):
                net.eval() 
                
                #print(net(X.to(device)).argmax(dim=1))
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().cpu().item()
                net.train() 
            else: 
                if('is_training' in net.__code__.co_varnames): 
                    
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n
    
def train(net, train_iter, test_iter, batch_size, optimizer, num_epochs, device, evaluate_accuracy):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X , y= X.to(device), y.to(device)
            #print(X.type(),X.size())
            y=y.view(1,-1)[0]
            y=y.type(torch.LongTensor)
            y = y.to(device)
            #print(y.type(),y.size())
            y_hat = net(X)
            
            #print(y_hat.type(),y_hat.size())
            l = loss(y_hat, y)
            optimizer.zero_grad()
            #print('a',net.conv1.weight.grad)
            l.backward()
            
            #print('b',net.conv1.weight.grad)
            
            optimizer.step()
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

    return net, optimizer
        

