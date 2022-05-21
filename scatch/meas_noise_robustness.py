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

DEFAULT_SIGMAS = (0.12, 0.25, 0.50, 1.00, 1.25)

def meas_noise_robustness(nn_model, test_data, test_targets, MC_itr=100, alpha=0.001, sigmas=DEFAULT_SIGMAS):
    # Get output shape by passing in one test image
    nn_out_size = nn_model(test_data).size() # num_images x num_classes
    # print(f'nn_model output size = {nn_out_size}')
    # print(f'targets output size = {test_targets.size()}')

    R_vals = torch.empty((len(sigmas), nn_out_size[0])) # num_sigmas x num_images
    # For each noise level sigma:
    for sigma_idx, sigma in enumerate(sigmas):
        # For N Monte-Carlo iterations:
        print(f'Sigma = {sigma}:')        
        mc_y_pred = torch.zeros(nn_out_size, dtype=torch.int16) # num_images x num_classes
        for itr in tqdm.tqdm(range(MC_itr), desc='AWGN samples'):
            # pre-process test_data with awgn
            test_data_noised = test_data.detach()
            test_data_noised.add_(sigma*torch.randn(test_data_noised.size()))
            y_pred = nn_model(test_data_noised) # make predictions
            # Save predictions            
            maxes = torch.argmax(y_pred, dim=1) # num_images x 1 (prediction per image)
            for idx in range(maxes.numel()):
                mc_y_pred[idx][maxes[idx].item()] += 1 # Add to prediction count for each image
        
        # # For each image, print probability of correct prediction
        # total_correct = 0
        # for idx in range(test_targets.numel()):
        #     total_correct += mc_y_pred[idx][test_targets[idx].item()].item()
        # percent_correct = float(total_correct)/(MC_itr*test_targets.numel())*100.0
        # print(f'Accuracy: {round(percent_correct, 2)}%')
        
        for idx in tqdm.tqdm(range(mc_y_pred.size()[0]), desc='Calculating radii'): # Compute and save R for every image
            max_cnt = torch.max(mc_y_pred.select(0, idx)).item()
            pa = lower_conf_bound(max_cnt, MC_itr, alpha)
            R = -1.0
            if pa > 0.5: R = sigma*NormalDist().inv_cdf(pa)
            R_vals[sigma_idx][idx] = R

        print()
    return R_vals # Return R for every noise level and image

@functools.lru_cache()
def lower_conf_bound_old(k, n, alpha):
    max_bnd = alpha**(1/n)
    if k==n:
        return max_bnd
    else:
        p = float(k)/float(n)
        z = NormalDist().inv_cdf(1-alpha) # aka "z-score"
        ndist_bnd = p - z*sqrt( p*(1-p) / n )
        return min(max_bnd, ndist_bnd)
        # Due to some bug, this doesn't work at all!
        #return proportion_confint(k, n, alpha=alpha*2, method='binom_test')[0]

@functools.lru_cache()
def lower_conf_bound(k, n, alpha):
    # Binary search to invert binomial tests. A bit ridiculous...
    BIN_SEARCH_ITRS = 24 # Enough for 32-bit floating-point mantissa
    prob = 0.5
    for i in range(BIN_SEARCH_ITRS):
        z = binomtest(k, n, p=prob, alternative='greater').pvalue
        if z < alpha: prob += (1/(2**(i+1)))
        else:         prob -= (1/(2**(i+1)))
        #print(f'prob = {prob}, resulting p-value = {z}')
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
    test_loader = torch.utils.data.DataLoader(torch_test_dataset, batch_size = len(torch_test_dataset), shuffle=True)
    test_data, test_targets = get_next_data_batch(test_loader, device)
    print('Test data loaded.')
    print()

    # Estimate robustness
    custom_sigmas=(0.25, 0.50, 1.00, 1.25)
    result = meas_noise_robustness(model, test_data, test_targets,
                                   MC_itr=200, alpha=0.001, sigmas=custom_sigmas)
    print('Evaluation complete!')
    torch.save(result, 'data/robustness_result.pt')

    # Compute proportion of images within robustness radius
    rs = np.linspace(0.0, 3.0, num=1000)
    Rs_vals = torch.empty((len(rs), len(custom_sigmas)))
    for i, r in enumerate(rs):
        Rs_vals[i] = (result>r).sum(dim=1)/(result.size()[1])

    # Graph accuracy vs radius
    plt.figure()
    for i, s in enumerate(custom_sigmas):
        plt.plot(rs, Rs_vals.select(dim=1, index=i), label=f'sigma = {s}')
    
    plt.title('Certified Accuracy vs radius of LeNet on MNIST test set')
    plt.xlabel('Radius')
    plt.ylim([0.0, 1.0])
    plt.ylabel('Certified Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':        
    sys.path.insert(0, './scatch')
    from MNIST_CNN_script import *
    main()