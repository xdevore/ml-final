

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os, sys
from matplotlib import pyplot as plt
#%matplotlib inline
import time
import pickle
import pandas
import gzip

import seaborn as sns


num_cca_trials = 5

def positivedef_matrix_sqrt(array):

  w, v = np.linalg.eigh(array)
  #  A - np.dot(v, np.dot(np.diag(w), v.T))
  wsqrt = np.sqrt(w)
  sqrtarray = np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))
  return sqrtarray


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon):


  x_diag = np.abs(np.diagonal(sigma_xx))
  y_diag = np.abs(np.diagonal(sigma_yy))
  x_idxs = (x_diag >= epsilon)
  y_idxs = (y_diag >= epsilon)

  sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
  sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
  sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
  sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

  return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop,
          x_idxs, y_idxs)


def compute_ccas(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon,
                 verbose=True):


  (sigma_xx, sigma_xy, sigma_yx, sigma_yy,
   x_idxs, y_idxs) = remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon)

  numx = sigma_xx.shape[0]
  numy = sigma_yy.shape[0]

  if numx == 0 or numy == 0:
    return ([0, 0, 0], [0, 0, 0], np.zeros_like(sigma_xx),
            np.zeros_like(sigma_yy), x_idxs, y_idxs)

  if verbose:
    print("adding eps to diagonal and taking inverse")
  sigma_xx += epsilon * np.eye(numx)
  sigma_yy += epsilon * np.eye(numy)
  inv_xx = np.linalg.pinv(sigma_xx)
  inv_yy = np.linalg.pinv(sigma_yy)

  if verbose:
    print("taking square root")
  invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
  invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

  if verbose:
    print("dot products...")
  arr = np.dot(invsqrt_xx, np.dot(sigma_xy, invsqrt_yy))

  if verbose:
    print("trying to take final svd")
  u, s, v = np.linalg.svd(arr)

  if verbose:
    print("computed everything!")

  return [u, np.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold(array, threshold):

  assert (threshold >= 0) and (threshold <= 1), "print incorrect threshold"

  for i in range(len(array)):
    if np.sum(array[:i])/np.sum(array) >= threshold:
      return i


def create_zero_dict(compute_dirns, dimension):

  return_dict = {}
  return_dict["mean"] = (np.asarray(0), np.asarray(0))
  return_dict["sum"] = (np.asarray(0), np.asarray(0))
  return_dict["cca_coef1"] = np.asarray(0)
  return_dict["cca_coef2"] = np.asarray(0)
  return_dict["idx1"] = 0
  return_dict["idx2"] = 0

  if compute_dirns:
    return_dict["cca_dirns1"] = np.zeros((1, dimension))
    return_dict["cca_dirns2"] = np.zeros((1, dimension))

  return return_dict


def get_cca_similarity(acts1, acts2, epsilon=0., threshold=0.98,
                       compute_coefs=True,
                       compute_dirns=False,
                       verbose=True):


  # assert dimensionality equal
  assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
  # check that acts1, acts2 are transposition
  assert acts1.shape[0] < acts1.shape[1], ("input must be number of neurons"
                                           "by datapoints")
  return_dict = {}

  # compute covariance with numpy function for extra stability
  numx = acts1.shape[0]
  numy = acts2.shape[0]

  covariance = np.cov(acts1, acts2)
  sigmaxx = covariance[:numx, :numx]
  sigmaxy = covariance[:numx, numx:]
  sigmayx = covariance[numx:, :numx]
  sigmayy = covariance[numx:, numx:]

  # rescale covariance to make cca computation more stable
  xmax = np.max(np.abs(sigmaxx))
  ymax = np.max(np.abs(sigmayy))
  sigmaxx /= xmax
  sigmayy /= ymax
  sigmaxy /= np.sqrt(xmax * ymax)
  sigmayx /= np.sqrt(xmax * ymax)

  ([u, s, v], invsqrt_xx, invsqrt_yy,
   x_idxs, y_idxs) = compute_ccas(sigmaxx, sigmaxy, sigmayx, sigmayy,
                                  epsilon=epsilon,
                                  verbose=verbose)

  # if x_idxs or y_idxs is all false, return_dict has zero entries
  if (not np.any(x_idxs)) or (not np.any(y_idxs)):
    return create_zero_dict(compute_dirns, acts1.shape[1])

  if compute_coefs:

    # also compute full coefficients over all neurons
    x_mask = np.dot(x_idxs.reshape((-1, 1)), x_idxs.reshape((1, -1)))
    y_mask = np.dot(y_idxs.reshape((-1, 1)), y_idxs.reshape((1, -1)))

    return_dict["coef_x"] = u.T
    return_dict["invsqrt_xx"] = invsqrt_xx
    return_dict["full_coef_x"] = np.zeros((numx, numx))
    np.place(return_dict["full_coef_x"], x_mask,
             return_dict["coef_x"])
    return_dict["full_invsqrt_xx"] = np.zeros((numx, numx))
    np.place(return_dict["full_invsqrt_xx"], x_mask,
             return_dict["invsqrt_xx"])

    return_dict["coef_y"] = v
    return_dict["invsqrt_yy"] = invsqrt_yy
    return_dict["full_coef_y"] = np.zeros((numy, numy))
    np.place(return_dict["full_coef_y"], y_mask,
             return_dict["coef_y"])
    return_dict["full_invsqrt_yy"] = np.zeros((numy, numy))
    np.place(return_dict["full_invsqrt_yy"], y_mask,
             return_dict["invsqrt_yy"])

    # compute means
    neuron_means1 = np.mean(acts1, axis=1, keepdims=True)
    neuron_means2 = np.mean(acts2, axis=1, keepdims=True)
    return_dict["neuron_means1"] = neuron_means1
    return_dict["neuron_means2"] = neuron_means2

  if compute_dirns:
    # orthonormal directions that are CCA directions
    cca_dirns1 = np.dot(np.dot(return_dict["full_coef_x"],
                               return_dict["full_invsqrt_xx"]),
                        (acts1 - neuron_means1)) + neuron_means1
    cca_dirns2 = np.dot(np.dot(return_dict["full_coef_y"],
                               return_dict["full_invsqrt_yy"]),
                        (acts2 - neuron_means2)) + neuron_means2

  # get rid of trailing zeros in the cca coefficients
  idx1 = sum_threshold(s, threshold)
  idx2 = sum_threshold(s, threshold)

  return_dict["cca_coef1"] = s
  return_dict["cca_coef2"] = s
  return_dict["x_idxs"] = x_idxs
  return_dict["y_idxs"] = y_idxs
  # summary statistics
  return_dict["mean"] = (np.mean(s[:idx1]), np.mean(s[:idx2]))
  return_dict["sum"] = (np.sum(s), np.sum(s))

  if compute_dirns:
    return_dict["cca_dirns1"] = cca_dirns1
    return_dict["cca_dirns2"] = cca_dirns2

  return return_dict


def robust_cca_similarity(acts1, acts2, threshold=0.98, epsilon=1e-6,
                          compute_dirns=True):


  for trial in range(num_cca_trials):
    try:
      return_dict = get_cca_similarity(acts1, acts2, threshold, compute_dirns)
    except np.LinAlgError:
      acts1 = acts1*1e-1 + np.random.normal(size=acts1.shape)*epsilon
      acts2 = acts2*1e-1 + np.random.normal(size=acts1.shape)*epsilon
      if trial + 1 == num_cca_trials:
        raise

  return return_dict



#------------------------------------------our code---------------------------------



# sys.path.append("..")
# import cca_core

def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig("rap_and_rock.png")
    plt.show()

def cutoff(s):
    total_variance = np.sum(s)
    desired_variance_fraction = 0.99
    cumulative_variance = 0
    cutoff_index = -1
    for idx, singular_value in enumerate(s):
        cumulative_variance += singular_value
        if cumulative_variance / total_variance >= desired_variance_fraction:
            cutoff_index = idx
            break
    return cutoff_index

def load_npy_file(filename):
    try:
        array = np.load(filename)
        print(f"Loaded {filename} successfully")
        return array
    except Exception as e:
        print(f"Error loading {filename}: {e}")


def process_data(acts1,acts2):


    # Reshape activations
    num_datapoints, h, w, channels = acts1.shape
    f_acts1 = acts1.reshape((num_datapoints*h*w, channels))
    f_acts1 = f_acts1.T[:,::2]
    num_datapoints, h, w, channels = acts2.shape
    f_acts2 = acts2.reshape((num_datapoints*h*w, channels))
    f_acts2= f_acts2.T[:,::2]

    print(f_acts1.shape, f_acts2.shape)
    return f_acts1, f_acts2

def get_SVCCA(f_acts1,f_acts2):
    # Mean subtract reshaped activations
    cacts1 = f_acts1 - np.mean(f_acts1, axis=1, keepdims=True)
    cacts2 = f_acts2 - np.mean(f_acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    # Use this to get the cutoff point forbest_model.h our data - 99% variance
    cutoff_acts1 = cutoff(s1)
    cutoff_acts2 = cutoff(s2)
    print(len(s1))
    print("cutoofffff 1",cutoff_acts1)
    print("cutoff2",cutoff_acts2)

    # Compute svacts1 and svacts2
    svacts1 = np.dot(np.diag(s1[:cutoff_acts1]), V1[:cutoff_acts1])
    svacts2 = np.dot(np.diag(s2[:cutoff_acts2]), V2[:cutoff_acts2])
    print(svacts1.shape)
    svcca_results = get_cca_similarity(svacts1, svacts2, epsilon=1e-10, compute_dirns=True, verbose=False)
    # Compute SVCCA similarity
    return svcca_results



def graph_top_svcca_directions(svcca_results, top_n=3):
    # Get the top_n CCA coefficients
    print(svcca_results["cca_coef1"].shape)
    top_n_coeffs = svcca_results["cca_coef1"][:top_n]

    # Get the corresponding CCA directions for both activation sets
    top_n_dirns1 = svcca_results["cca_dirns1"][:, :top_n]
    top_n_dirns2 = svcca_results["cca_dirns2"][:, :top_n]

    # Plot the top_n CCA directions for both activation sets
    for i in range(top_n):
        plt.figure()
        plt.plot(top_n_dirns1[:, i], label=f"Direction {i+1} - Acts1")
        plt.plot(top_n_dirns2[:, i], label=f"Direction {i+1} - Acts2")
        plt.xlabel("Data Index")
        plt.ylabel("Neuron Activation")
        plt.legend()
        plt.title(f"Top {i+1} SVCCA Direction")
        plt.grid()
        plt.show()

def run_tests_all(files):
    num_activations = len(files)
    svcca_coeffs = np.zeros((num_activations, num_activations))

    for i in range(num_activations):
        activations1 = load_npy_file(files[i])
        accmid =activations1
        for j in range(i, num_activations):

            activations2 = load_npy_file(files[j])
            print("acts 1 shape", accmid.shape, activations2.shape)
            activations1, activations2 = process_data(accmid,activations2)
            coeff = get_SVCCA(activations1,activations2)
            svcca_coeffs[i, j] = np.mean(coeff["cca_coef1"])
            svcca_coeffs[j, i] = np.mean(coeff["cca_coef1"])
    middle_parts = []
    for file in files:
        parts = file.split("_")
        middle_part = parts[2].split(".")[0]
        middle_parts.append(middle_part)
# Create a heatmap of the SVCCA coefficients
    ax = sns.heatmap(svcca_coeffs, annot=True, cmap="coolwarm", xticklabels=middle_parts, yticklabels=middle_parts)
    plt.title("SVCCA Coefficients Heatmap")
    plt.xlabel("Activations")
    plt.ylabel("Activations")
    plt.show()
    return svcca_coeffs


file_names = ["activations_matrix_house.npy","activations_matrix_rap.npy","activations_matrix_rock.npy","activations_matrix_rock1.npy"]
filename1 = "activations_matrix_rock.npy"
filename2 = "activations_matrix_rock1.npy"

act1 = load_npy_file(filename1)
act2 = load_npy_file(filename2)


f_acts1, f_acts2 = process_data(act1,act2)

svcca_results = get_SVCCA(f_acts1,f_acts2)
#
# # Load .npy files as NumPy arrays
#
#
#
# print("SVCCA shape", svcca_results["cca_dirns1"].shape)
#
# graph_top_svcca_directions(svcca_results)


run_tests_all(file_names)


print("MNIST", np.mean(svcca_results["cca_coef1"]))
#
# # Plot the results
# _plot_helper(svcca_results["cca_coef1"], "CCA Coef idx", "CCA coef value")
