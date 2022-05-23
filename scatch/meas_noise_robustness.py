import torch
import sys
from math import sqrt
import numpy as np
from statistics import NormalDist
from scipy.stats import binomtest
import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
import functools

device = 'cpu'

DEFAULT_SIGMAS = (0.12, 0.25, 0.50, 1.00, 1.25)

@torch.no_grad()
def meas_noise_robustness(nn_model, dataloader, MC_itr=100, alpha=0.001, sigmas=DEFAULT_SIGMAS):
    # Get output shape
    nn_out_size = (len(dataloader.dataset), len(dataloader.dataset.classes))
    batch_size = dataloader.batch_size # Get number of batches

    R_vals = torch.empty((len(sigmas), nn_out_size[0])) # num_sigmas x num_images
    # For each noise level sigma:
    for sigma_idx, sigma in enumerate(sigmas):
        # For N Monte-Carlo iterations:
        print(f'Sigma = {sigma}:')
        mc_y_pred = torch.zeros(nn_out_size, dtype=torch.int32) # num_images x num_classes
        for itr in tqdm.tqdm(range(MC_itr), desc='AWGN samples'):
            # For each batch
            for batch_idx, (data, lab) in enumerate(dataloader):
                data = data.to(device)
                # pre-process test_data with awgn
                test_data_noised = data.detach()
                test_data_noised.add_(sigma**2*torch.randn(test_data_noised.size()))
                y_pred = nn_model(test_data_noised) # make predictions
                # Save predictions
                maxes = torch.argmax(y_pred, dim=1) # num_images x 1 (prediction per image)
                # print(f'maxes shape = {maxes.shape}')
                for idx in range(maxes.numel()):
                    # print(f'len = {batch_size}, b_idx = {batch_idx}, idx = {idx}')
                    mc_y_pred[batch_size*batch_idx+idx][maxes[idx].item()] += 1 # Add to prediction count for each image

        # Print the total prediction accuracy
        total_correct = 0
        targs = dataloader.dataset.targets
        for idx in range(targs.numel()):
            total_correct += mc_y_pred[idx][targs[idx].item()].item()
        percent_correct = float(total_correct)/(MC_itr*targs.numel())*100.0
        print(f'Prediction Accuracy (under AWGN): {round(percent_correct, 2)}%')
        
        for idx in tqdm.tqdm(range(mc_y_pred.size()[0]), desc='Calculating radii'): # Compute and save R for every image
            max_cnt = torch.max(mc_y_pred.select(0, idx)).item()
            pa = lower_conf_bound(max_cnt, MC_itr, alpha)
            R = sigma*NormalDist().inv_cdf(pa)
            R_vals[sigma_idx][idx] = R

        print()
    return R_vals # Return R for every noise level and image

def lower_conf_bound_old(k, n, alpha):
    max_bnd = alpha**(1/n)
    if k==n:
        return max_bnd
    else:
        p = float(k)/float(n)
        z = NormalDist().inv_cdf(1-alpha) # aka "z-score"
        ndist_bnd = p - z*sqrt( p*(1-p) / n )
        return min(max_bnd, ndist_bnd)

@functools.lru_cache() # Cache results for speed
def lower_conf_bound(k, n, alpha):
    # Binary search to invert binomial tests. A bit ridiculous...
    BIN_SEARCH_ITRS = 24 # Enough for 32-bit floating-point mantissa
    prob = 0.5
    for i in range(BIN_SEARCH_ITRS):
        z = binomtest(k, n, p=prob, alternative='greater').pvalue
        if z < alpha: prob += (1/(2**(i+1)))
        else:         prob -= (1/(2**(i+1)))
    prob -= (1/(2**(BIN_SEARCH_ITRS))) # Subtract the maximum search error
    return prob

# Demo using model from MNIST_CNN_script.py
def main():
    # Load saved model
    model = my_LeNet_Model({})
    model.load_state_dict(torch.load('./scatch/MNIST_CNN_MODEL.pt'))
    print('Model loaded.')

    # use full data for test set (no batches)
    dataFolder = os.path.dirname(script_folder_path) + '/data/'
    torch_test_dataset = torchvision.datasets.MNIST(dataFolder, train=False, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    test_loader = torch.utils.data.DataLoader(torch_test_dataset, 
                                              batch_size = 10000, 
                                              shuffle=False)
    print(f'Test data loaded, batch size = {test_loader.batch_size}')
    print()

    # Estimate robustness
    custom_sigmas = (0.25, 0.50, 1.00, 1.25)
    N = 200
    result = meas_noise_robustness(model, test_loader,
                                   MC_itr=N, alpha=0.001, sigmas=custom_sigmas)
    print('Evaluation complete!')
    torch.save(result, 'data/robustness_result.pt')

    # Compute proportion of images within robustness radius
    rs = np.linspace(0.0, 5.0, num=1000)
    Rs_vals = torch.empty((len(rs), len(custom_sigmas)))
    for i, r in enumerate(rs):
        Rs_vals[i] = (result>r).sum(dim=1)/(result.size()[1])

    # Graph accuracy vs radius
    plt.figure()
    for i, s in enumerate(custom_sigmas):
        plt.plot(rs, Rs_vals.select(dim=1, index=i), label=f'sigma = {s}')
    
    plt.title(f'Certified Accuracy vs radius on MNIST test set (N={N})')
    plt.xlabel('Radius')
    plt.ylim([0.0, 1.0])
    plt.ylabel('Certified Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':        
    sys.path.insert(0, './scatch')
    from MNIST_CNN_script import *
    if torch.cuda.is_available(): device = 'cuda' 
    print(f'Using device "{device}"')
    main()