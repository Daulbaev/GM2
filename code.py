import numpy as np
import graph_cut
import time
import sys
import matplotlib.pylab as plt

def get_3d_of_2d(A, x):
    """
    Parameters
        ----------
        A : 3-dimensional np.array
        x : 2-dimensional np.array
    Returns
        -------
        y : 2-dimensional np.array such that y[i, j] = A[i, j, x[i, j]]
    """
    N, M, K = A.shape
    idx = np.vstack((np.indices((N, M)), x[np.newaxis, :, :]))
    y = A[idx[0, :, :], idx[1, :, :], idx[2, :, :]]
    return y


def my_compute_full_energy(x, unary, grayscale):
    N, M, K = unary.shape
    # Unary:
    E = np.sum(get_3d_of_2d(unary, x))
    # Paired:
    E += np.sum(np.abs(get_3d_of_2d(grayscale[:-1, :, :], x[:-1, :]) - 
           get_3d_of_2d(grayscale[:-1, :, :], x[1:, :])))
    E += np.sum(np.abs(get_3d_of_2d(grayscale[1:, :, :], x[:-1, :]) - 
           get_3d_of_2d(grayscale[1:, :, :], x[1:, :])))
    E += np.sum(np.abs(get_3d_of_2d(grayscale[:, :-1, :], x[:, :-1]) - 
           get_3d_of_2d(grayscale[:, :-1, :], x[:, 1:])))
    E += np.sum(np.abs(get_3d_of_2d(grayscale[:, 1:, :], x[:, :-1]) - 
           get_3d_of_2d(grayscale[:, 1:, :], x[:, 1:])))
    return E


def my_alphaExpansion(unary, grayscale, maxIter=500, display=False, numStart=1, randOrder=False):
    N, M, K = unary.shape
    best_labels, best_energy, best_time = 0, [np.inf], []
    for n_start in range(numStart):
        # random initial approximation for marking:
        x_new = np.random.randint(0, K, (N, M))
        cur_time, cur_energy = [], [np.inf]
        t = time.time()
        for n_iter in range(maxIter):
            if display:
                print(n_iter)
                sys.stdout.flush()
            labelsOrdering = np.random.permutation(K) if randOrder else np.arange(K)
            for alpha in labelsOrdering:
                x_old = x_new.copy()
                x_new, E = my_alphaExpansionStep(x_old, alpha, unary, grayscale)
            # energy = my_compute_full_energy(x_new, unary, grayscale)
            # if np.abs(E - energy) < 1e-6:
            #     raise RuntimeError('Energies doesn't match!')
            cur_time.append(time.time() - t)
            t = time.time()
            if cur_energy[-1] - E <= -1e-6:
                raise RuntimeError('Energy must decrease!')
            cur_energy.append(E)
            if np.all(x_new == x_old):
                break
        if cur_energy[-1] < best_energy[-1]:
            best_labels = x_new.copy()
            best_energy = cur_energy
            best_time = cur_time
    return best_labels, np.array(best_energy[1:]), np.array(best_time)


def my_alphaExpansionStep(x_old, alpha, unary, grayscale):
    N, M, K = unary.shape
    idx = np.vstack((np.indices((N, M)), x_old[np.newaxis, :, :]))
    bin_unary = np.dstack((unary[idx[0, :, :], idx[1, :, :], idx[2, :, :]], unary[:, :, alpha]))
    # vert_paired.shape --- (N - 1) × M × 2 × 2
    # hor_paired.shape --- N × (M - 1) × 2 × 2
    vert_paired = np.zeros((N - 1, M, 2, 2))
    hor_paired = np.zeros((N, M - 1, 2, 2))
    # phi(0, 0):
    vert_paired[:, :, 0, 0] = np.abs(get_3d_of_2d(grayscale[:-1, :, :], x_old[:-1, :]) - 
                                    get_3d_of_2d(grayscale[:-1, :, :], x_old[1:, :]))
    vert_paired[:, :, 0, 0] += np.abs(get_3d_of_2d(grayscale[1:, :, :], x_old[:-1, :]) - 
                                    get_3d_of_2d(grayscale[1:, :, :], x_old[1:, :]))
    hor_paired[:, :, 0, 0] = np.abs(get_3d_of_2d(grayscale[:, :-1, :], x_old[:, :-1]) - 
                                    get_3d_of_2d(grayscale[:, :-1, :], x_old[:, 1:]))
    hor_paired[:, :, 0, 0] += np.abs(get_3d_of_2d(grayscale[:, 1:, :], x_old[:, :-1]) - 
                                    get_3d_of_2d(grayscale[:, 1:, :], x_old[:, 1:]))
    # phi(1, 1) equals zero
    # phi(0, 1):
    vert_paired[:, :, 0, 1] = np.abs(get_3d_of_2d(grayscale[:-1, :, :], x_old[:-1, :]) - 
                                    grayscale[:-1, :, alpha])
    vert_paired[:, :, 0, 1] += np.abs(get_3d_of_2d(grayscale[1:, :, :], x_old[:-1, :]) - 
                                    grayscale[1:, :, alpha])
    hor_paired[:, :, 0, 1] = np.abs(get_3d_of_2d(grayscale[:, :-1, :], x_old[:, :-1]) - 
                                    grayscale[:, :-1, alpha])
    hor_paired[:, :, 0, 1] += np.abs(get_3d_of_2d(grayscale[:, 1:, :], x_old[:, :-1]) - 
                                    grayscale[:, 1:, alpha])
    # phi(1, 0):
    vert_paired[:, :, 1, 0] = np.abs(grayscale[:-1, :, alpha] - 
                                    get_3d_of_2d(grayscale[:-1, :, :], x_old[1:, :]))
    vert_paired[:, :, 1, 0] += np.abs(grayscale[1:, :, alpha] - 
                                    get_3d_of_2d(grayscale[1:, :, :], x_old[1:, :]))
    hor_paired[:, :, 1, 0] = np.abs(grayscale[:, :-1, alpha] - 
                                    get_3d_of_2d(grayscale[:, :-1, :], x_old[:, 1:]))
    hor_paired[:, :, 1, 0] += np.abs(grayscale[:, 1:, alpha] - 
                                    get_3d_of_2d(grayscale[:, 1:, :], x_old[:, 1:]))
    # Reparametrization:
    theta_0 = 0
    bin_unary[:-1, :, 0] += vert_paired[:, :, 0, 0]
    bin_unary[:, :-1, 0] += hor_paired[:, :, 0, 0]
    bin_unary[1:, :, 1] += vert_paired[:, :, 0, 1] - vert_paired[:, :, 0, 0]
    bin_unary[:, 1:, 1] += hor_paired[:, :, 0, 1] - hor_paired[:, :, 0, 0]
    bin_unary[:-1, :, 1] += vert_paired[:, :, 0, 0] - vert_paired[:, :, 0, 1]
    bin_unary[:, :-1, 1] += hor_paired[:, :, 0, 0] - hor_paired[:, :, 0, 1]
    
    vert_paired[:, :, 1, 0] += vert_paired[:, :, 0, 1] - vert_paired[:, :, 0, 0]
    vert_paired[:, :, 0, 0] = vert_paired[:, :, 0, 1] = 0
    
    hor_paired[:, :, 1, 0] += hor_paired[:, :, 0, 1] - hor_paired[:, :, 0, 0]
    hor_paired[:, :, 0, 0] = hor_paired[:, :, 0, 1] = 0
    
    minimum = np.min(bin_unary, axis=2)[:, :, np.newaxis]
    theta_0 += np.sum(minimum)
    bin_unary -= minimum
    hor_paired[:, :, 1, 0][hor_paired[:, :, 1, 0] < 0] = 0
    vert_paired[:, :, 1, 0][vert_paired[:, :, 1, 0] < 0] = 0
    # GraphCut:
    terminal_weights = bin_unary.reshape(-1, 2)[:, ::-1]
    idx = np.arange(N * M).reshape(N, M)
    vert_weights = np.hstack((
        np.dstack((idx[:-1, :], idx[1:, :])).reshape(-1, 2), 
        np.zeros(((N - 1) * M, 1)),
        vert_paired[:, :, 1, 0].reshape(-1, 1)
    ))
    hor_weights = np.hstack((
        np.dstack((idx[:, :-1], idx[:, 1:])).reshape(-1, 2), 
        np.zeros((N * (M - 1), 1)),
        hor_paired[:, :, 1, 0].reshape(-1, 1)
    ))
    edge_weights = np.vstack((hor_weights, vert_weights))
    cut, labels = graph_cut.graph_cut(terminal_weights, edge_weights)
    cut += theta_0
    x = x_old.copy()
    x[labels.reshape(N, M) == 1] = alpha
    return x, cut

def my_stichImages(images, seeds, return_energy_and_time=False):
    constant = 1000
    N, M = images[0].shape[:2]
    K = len(images)
    grayscale = np.empty((N, M, K))
    unary = np.full((N, M, K), constant, dtype=float)
    for idx, img in enumerate(images):
        grayscale[:, :, idx] = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    for idx, seed in enumerate(seeds):
        unary[:, :, idx][seed] = 0
    resultMask, enery, t = my_alphaExpansion(unary, grayscale, maxIter=2, display=True)
    resultImage = np.zeros_like(images[0])
    for idx, img in enumerate(images):
        resultImage[resultMask == idx, :] = images[idx][resultMask == idx, :]
    if return_energy_and_time:
        return resultImage, resultMask, enery, t 
    return resultImage, resultMask

def alphaExpansionGridPotts(unary, vertC, horC, metric, maxIter=500, 
                            display=False, numStart=1, randOrder=False):
    N, M, K = unary.shape
    best_labels, best_energy, best_time = 0, [np.inf], []
    for n_start in range(numStart):
        # random initial approximation for marking:
        x_new = np.random.randint(0, K, (N, M))
        cur_time, cur_energy = [], [np.inf]
        t = time.time()
        for n_iter in range(maxIter):
            if display:
                print(n_iter)
                sys.stdout.flush()
            labelsOrdering = np.random.permutation(K) if randOrder else np.arange(K)
            for alpha in labelsOrdering:
                x_old = x_new.copy()
                x_new, E = step_alphaExpansionGridPotts(x_old, alpha, unary, vertC, horC, metric)
            cur_time.append(time.time() - t)
            t = time.time()
            if cur_energy[-1] - E <= -1e-6:
                raise RuntimeError('Energy must decrease!')
            cur_energy.append(E)
            if np.all(x_new == x_old):
                break
        if cur_energy[-1] < best_energy[-1]:
            best_labels = x_new.copy()
            best_energy = cur_energy
            best_time = cur_time
    return best_labels, np.array(best_energy[1:]), np.array(best_time)

def step_alphaExpansionGridPotts(x_old, alpha, unary, vertC, horC, metric):
    N, M, K = unary.shape
    idx = np.vstack((np.indices((N, M)), x_old[np.newaxis, :, :]))
    bin_unary = np.dstack((unary[idx[0, :, :], idx[1, :, :], idx[2, :, :]], unary[:, :, alpha]))
    vert_paired = np.zeros((N - 1, M, 2, 2))
    hor_paired = np.zeros((N, M - 1, 2, 2))
    # phi(0, 0):
    vert_paired[:, :, 0, 0] = vertC * metric[x_old[:-1, :], x_old[1:, :]]
    hor_paired[:, :, 0, 0] = horC * metric[x_old[:, :-1], x_old[:, 1:]]
    # phi(1, 1):
    vert_paired[:, :, 1, 1] = vertC * metric[alpha, alpha]
    hor_paired[:, :, 1, 1] = horC * metric[alpha, alpha]
    # phi(0, 1):
    vert_paired[:, :, 0, 1] = vertC * metric[x_old[:-1, :], alpha]
    hor_paired[:, :, 0, 1] = horC * metric[x_old[:, :-1], alpha]
    # phi(1, 0):
    vert_paired[:, :, 1, 0] = vertC * metric[alpha, x_old[1:, :]]
    hor_paired[:, :, 1, 0] = horC * metric[alpha, x_old[:, 1:]]
    # Reparametrization:
    theta_0 = 0
    bin_unary[:-1, :, 0] += vert_paired[:, :, 0, 0]
    bin_unary[:, :-1, 0] += hor_paired[:, :, 0, 0]
    bin_unary[1:, :, 1] += vert_paired[:, :, 0, 1] - vert_paired[:, :, 0, 0]
    bin_unary[:, 1:, 1] += hor_paired[:, :, 0, 1] - hor_paired[:, :, 0, 0]
    bin_unary[:-1, :, 1] += vert_paired[:, :, 0, 0] - vert_paired[:, :, 0, 1]
    bin_unary[:, :-1, 1] += hor_paired[:, :, 0, 0] - hor_paired[:, :, 0, 1]
    vert_paired[:, :, 1, 0] += vert_paired[:, :, 0, 1] - vert_paired[:, :, 0, 0]
    vert_paired[:, :, 0, 0] = vert_paired[:, :, 0, 1] = 0
    hor_paired[:, :, 1, 0] += hor_paired[:, :, 0, 1] - hor_paired[:, :, 0, 0]
    hor_paired[:, :, 0, 0] = hor_paired[:, :, 0, 1] = 0
    minimum = np.min(bin_unary, axis=2)[:, :, np.newaxis]
    theta_0 += np.sum(minimum)
    bin_unary -= minimum
    hor_paired[:, :, 1, 0][hor_paired[:, :, 1, 0] < 0] = 0
    vert_paired[:, :, 1, 0][vert_paired[:, :, 1, 0] < 0] = 0
    # GraphCut:
    terminal_weights = bin_unary.reshape(-1, 2)[:, ::-1]
    idx = np.arange(N * M).reshape(N, M)
    vert_weights = np.hstack((
        np.dstack((idx[:-1, :], idx[1:, :])).reshape(-1, 2), 
        np.zeros(((N - 1) * M, 1)),
        vert_paired[:, :, 1, 0].reshape(-1, 1)
    ))
    hor_weights = np.hstack((
        np.dstack((idx[:, :-1], idx[:, 1:])).reshape(-1, 2), 
        np.zeros((N * (M - 1), 1)),
        hor_paired[:, :, 1, 0].reshape(-1, 1)
    ))
    edge_weights = np.vstack((hor_weights, vert_weights))
    cut, labels = graph_cut.graph_cut(terminal_weights, edge_weights)
    cut += theta_0
    x = x_old.copy()
    x[labels.reshape(N, M) == 1] = alpha
    return x, cut

def stichImages(images, seeds, return_energy_and_time=False):
    constant = 1000
    N, M = images[0].shape[:2]
    K = len(images)
    unary = np.full((N, M, K), constant, dtype=float)
    grayscale = np.empty((N, M, K))
    for idx, img in enumerate(images):
        grayscale[:, :, idx] = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    for idx, seed in enumerate(seeds):
        unary[:, :, idx][seed] = 0
    metric = 1 - np.eye(K)
    vertC = 1 - np.exp(-np.sum(np.abs(grayscale[-1:, :, :] - grayscale[1:, :, :]), axis=2))
    horC = 1 - np.exp(-np.sum(np.abs(grayscale[:, -1:, :] - grayscale[:, 1:, :]), axis=2))
    
    resultMask, enery, t = alphaExpansionGridPotts(unary, vertC, horC, metric, maxIter=20, display=True)
    resultImage = np.zeros_like(images[0])
    for idx, img in enumerate(images):
        resultImage[resultMask == idx, :] = images[idx][resultMask == idx, :]
    if return_energy_and_time:
        return resultImage, resultMask, enery, t 
    return resultImage, resultMask
