import torch
import sys
import tqdm

def meas_noise_robustness(nn_model, test_data, test_targets, MC_itr=1000):
    # Get output shape by passing in one test image
    nn_out_size = nn_model(test_data).size() # num_images x num_classes
    # print(f'nn_model output size = {nn_out_size}')
    # print(f'targets output size = {test_targets.size()}')

    # For each noise level sigma:
    for sigma in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        # For N Monte-Carlo iterations:
        print(f'Sigma = {sigma}:')        
        mc_y_pred = torch.zeros(nn_out_size, dtype=torch.int16) # num_images x num_classes
        for itr in tqdm.trange(MC_itr):
            # pre-process test_data with awgn
            test_data_noised = test_data.detach()
            test_data_noised.add_(sigma*torch.randn(test_data.size(), requires_grad=False))
            y_pred = nn_model(test_data_noised) # make predictions
            # Save predictions            
            maxes = torch.argmax(y_pred, dim=1) # num_images x 1 (prediction per image)
            for idx in range(maxes.numel()):
                mc_y_pred[idx][maxes[idx].item()] += 1 # Add to prediction count for each image
        # For each image, find probability of correct prediction        
        #print(f'max = {mc_y_pred.max()}, min = {mc_y_pred.min()}')
        total_correct = 0
        for idx in range(test_targets.numel()):
            total_correct += mc_y_pred[idx][test_targets[idx].item()].item()
        percent_correct = float(total_correct)/(MC_itr*test_targets.numel())*100.0
        print(f'Accuracy: {round(percent_correct, 2)}%')
        # For each image, find probability of MOST LIKLEY incorrect prediction
        # Compute and save R for every image
        print()
    return 0

# Demo using model from MNIST_CNN_script.py
def main():
    model = my_LeNet_Model({})
    model.load_state_dict(torch.load('./scatch/MNIST_CNN_MODEL.pt'))
    print('Model loaded.')

    # use full data for test set (no batches)
    dataFolder = os.path.dirname(script_folder_path) + '/data/'
    torch_test_dataset = torchvision.datasets.MNIST(dataFolder, train=False, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = torch.utils.data.DataLoader(torch_test_dataset, batch_size = len(torch_test_dataset), shuffle=True)
    test_data, test_targets = get_next_data_batch(test_loader, device)
    print('Test data loaded.')

    print()
    meas_noise_robustness(model, test_data, test_targets, MC_itr=100)
    print('Evaluation complete!')

if __name__ == '__main__':        
    sys.path.insert(0, './scatch')
    from MNIST_CNN_script import *
    main()