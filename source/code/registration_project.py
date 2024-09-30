"""
Project code for image registration topics.
"""

import numpy as np
from statistics import median
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output


def rigid_reg_cc_demo(Img1,Img2,imshow=False):

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread(Img1)
    Im = plt.imread(Img2)

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    x = np.array([0., 0., 0.])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)

    # the learning rate
    mu = 0.001

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)
    if imshow:
        fig = plt.figure(figsize=(14,6))

        # fixed and moving image, and parameters
        ax1 = fig.add_subplot(121)

        # fixed image
        im1 = ax1.imshow(I)
        # moving image
        im2 = ax1.imshow(I, alpha=0.7)
        # parameters
        txt = ax1.text(0.3, 0.95,
            np.array2string(x, precision=5, floatmode='fixed'),
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
            transform=ax1.transAxes)

        # 'learning' curve
        ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

        learning_curve, = ax2.plot(iterations, similarity, lw=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Similarity')
        ax2.grid()

    # perform 'num_iter' gradient ascent updates
    past_s = [0,0,0,0,0]
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True)

        if imshow:
            clear_output(wait = True)

            # update moving image and parameters
            im2.set_data(Im_t)
            txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))
            # update 'learning' curve
            similarity[k] = S
            learning_curve.set_ydata(similarity)

            display(fig)

        # update 'learning' curve
        similarity[k] = S
        
        # End if: similarity is high, or if stability has been reached
        past_s.append(S)
        del past_s[0]
        var = np.var(past_s)
        if k > 5 and S > 0.99:
            break
        if k > 5 and var < 1e-7:
            break

    print(f'Final similarity of ca. {S}')

def affine_reg_cc_demo(img1,img2,imshow=False,num_iter=200, learning_rate=0.001):

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread(img1)
    Im = plt.imread(img2)

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    x = np.array([0., 1., 1., 0., 0. ,0. ,0.])
    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    fun = lambda x: reg.affine_corr(I, Im, x, return_transform=False)
    # the learning rate
    mu = learning_rate

    # number of iterations
    num_iter = num_iter

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)
    if imshow:
        fig = plt.figure(figsize=(14,6))

        # fixed and moving image, and parameters
        ax1 = fig.add_subplot(121)

        # fixed image
        im1 = ax1.imshow(I)
        # moving image
        im2 = ax1.imshow(I, alpha=0.7)
        # parameters
        txt = ax1.text(0.3, 0.95,
            np.array2string(x, precision=5, floatmode='fixed'),
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
            transform=ax1.transAxes)

        # 'learning' curve
        ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

        learning_curve, = ax2.plot(iterations, similarity, lw=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Similarity')
        ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):
        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu
        
        # for visualization of the result
        S, Im_t, _ = reg.affine_corr(I, Im, x, return_transform=True)
        if imshow:
            clear_output(wait = True)

            # update moving image and parameters
            im2.set_data(Im_t)
            txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

            # update 'learning' curve
            similarity[k] = S
            learning_curve.set_ydata(similarity)

            display(fig)
        similarity[k] = S  
    print(f'Final similarity of ca. {S}')
    return similarity

def affine_reg_mi_demo(img1,img2,imshow=False,num_iter=200, learning_rate=0.001):

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread(img1)
    Im = plt.imread(img2)

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    x = np.array([0., 1., 1., 0., 0. ,0. ,0.])
    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    fun = lambda x: reg.affine_mi(I, Im, x, return_transform=False)
    # the learning rate
    mu = learning_rate

    # number of iterations
    num_iter = num_iter

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)
    if imshow:
        fig = plt.figure(figsize=(14,6))

        # fixed and moving image, and parameters
        ax1 = fig.add_subplot(121)

        # fixed image
        im1 = ax1.imshow(I)
        # moving image
        im2 = ax1.imshow(I, alpha=0.7)
        # parameters
        txt = ax1.text(0.3, 0.95,
            np.array2string(x, precision=5, floatmode='fixed'),
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
            transform=ax1.transAxes)

        # 'learning' curve
        ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

        learning_curve, = ax2.plot(iterations, similarity, lw=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Similarity')
        ax2.grid()

    past_s = [0,0,0,0,0]
    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):
        
        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu
        
        # for visualization of the result
        S, Im_t, _ = reg.affine_mi(I, Im, x, return_transform=True)

        if imshow:
            clear_output(wait = True)

            # update moving image and parameters
            im2.set_data(Im_t)
            txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

            # update 'learning' curve
            similarity[k] = S
            learning_curve.set_ydata(similarity)

            display(fig)
        similarity[k] = S
        # End if: similarity is high, or if stability has been reached
        past_s.append(S)
        del past_s[0]
        var = np.var(past_s)
        if k > 5 and S > 0.99:
            break
        if k > 5 and var < 1e-7:
            break
    print(f'Final similarity of ca. {S} with variance {var}')

def absolute_error_histograms(J, I,imshow=False):
    A = []                    #empty list to add the differences between I and J in per index(value)
    G = 0
    for i in range(len(J)):    #Add de difference between J and I for every index point to the list A, for optional plots, and take the absolute of this difference(no negatives)
        A.append(abs(J[i]-I[i]))

    if imshow:
        xs = [x for x in range(len(A))]    #Plot the absolute error per value of J and I of the same index (optional)
        plt.plot(xs, A)
        plt.show()
    
    for i in A:                #Calculate the mean error by adding all the values of a and dividing this by the amount of elements in A
        G += A[i]
    G = G/len(A)
    return G                    #G will then be the absolute mean error
#absolute_error_histograms(J = [2,6,7,9,3,2,4], I = [3,8,5,7,6,4,9])
