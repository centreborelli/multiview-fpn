import numpy as np
import torch


def fwd_grad(img, eps=10**-20, batch=False):
    img_shape = img.shape
    if batch:
        grad_j = torch.diff(img, axis=2)
        grad_i = torch.diff(img, axis=1)
    else:
        grad_j = torch.diff(img, axis=1)
        grad_i = torch.diff(img, axis=0)
    
    # if grayscale
    if len(img_shape)==2:
        grad_j = torch.concatenate((grad_j, torch.zeros((img_shape[0], 1)).cuda()), axis=1)
        grad_i = torch.concatenate((grad_i, torch.zeros((1, img_shape[1])).cuda()), axis=0)
    elif len(img_shape)==3 and batch:
        grad_j = torch.concatenate((grad_j, torch.zeros((img_shape[0], img_shape[1], 1)).cuda()), axis=2)
        grad_i = torch.concatenate((grad_i, torch.zeros((img_shape[0], 1, img_shape[2])).cuda()), axis=1)
    elif len(img_shape)==3 and (not batch):
        grad_j = torch.concatenate((grad_j, torch.zeros((img_shape[0], 1, img_shape[2])).cuda()), axis=2)
        grad_i = torch.concatenate((grad_i, torch.zeros((1, img_shape[1], img_shape[2])).cuda()), axis=1)
    elif len(img_shape)==4:
        grad_j = torch.concatenate((grad_j, torch.zeros((img_shape[0], img_shape[1], 1, img_shape[3])).cuda()), axis=2)
        grad_i = torch.concatenate((grad_i, torch.zeros((img_shape[0], 1, img_shape[2], img_shape[3])).cuda()), axis=1)
    grad_j += eps
    grad_i += eps
    return grad_i, grad_j


def fwd_dgrad(img, eps=10 ** -10, batch=False):
    img_shape = img.shape
    if batch:
        grad_j = torch.diff(img, axis=2)
        grad_i = torch.diff(img, axis=1)

        grad_dl = grad_i[:, :, 1:] + grad_j[:, :-1, :]
        grad_dr = grad_i[:, :, :-1] - grad_j[:, :-1, :]
    else:
        grad_j = torch.diff(img, axis=1)
        grad_i = torch.diff(img, axis=0)

        grad_dl = grad_i[:, 1:] + grad_j[:-1, :]
        grad_dr = grad_i[:, :-1] - grad_j[:-1, :]
    grad_dl += eps
    grad_dr += eps
    return grad_dl, grad_dr


def TV(img, batch=True):
    grad_i, grad_j = fwd_grad(img, batch=batch)
    TV_term = torch.sqrt(grad_i**2 + grad_j**2).sum()
    return TV_term


def TV_2(img, batch=True):
    grad_i, grad_j = fwd_grad(img, batch=batch)
    TV_term = (grad_i**2 + grad_j**2).sum()
    return TV_term


def TVaniso(img, batch=True):
    grad_i, grad_j = fwd_grad(img, batch=batch)
    TV_term = (torch.abs(grad_i) + torch.abs(grad_j)).sum()
    return TV_term


def TV_diag(img, batch=True):
    grad_dl, grad_dr = fwd_dgrad(img, batch=batch)
    TV_term = torch.sqrt(grad_dl**2 + grad_dr**2).sum()
    return TV_term

def Div(vec_f):
    vec_fi, vec_fj = vec_f
    vec_f_shape = vec_fi.shape
    
    vec_fj[:, -1] = 0
    vec_fi[-1, :] = 0
    
    if len(vec_f_shape)==2:
        vec_fj = torch.concatenate((torch.zeros((vec_f_shape[0], 1)).cuda(), vec_fj), axis=1)
        vec_fi = torch.concatenate((torch.zeros((1, vec_f_shape[1])).cuda(), vec_fi), axis=0)
    else:
        vec_fj = torch.concatenate((torch.zeros((vec_f_shape[0], 1, vec_f_shape[2])).cuda(), vec_fj), axis=1)
        vec_fi = torch.concatenate((torch.zeros((1, vec_f_shape[1], vec_f_shape[2])).cuda(), vec_fi), axis=0)
        
    div_vec_j = torch.diff(vec_fj, axis=1)
    
    div_vec_i = torch.diff(vec_fi, axis=0)
    
    return div_vec_i + div_vec_j
    
    

def proj_F(hat_V):
    norm_tensor = (hat_V[:,0,:])**2 + (hat_V[:,1,:])**2
    norm_tensor = np.repeat(norm_tensor[:,None,:], 2, axis=1)
    V = hat_V / np.maximum(1, norm_tensor)
    return V
    
    
def proj_J(G, O, lambda_G, lambda_O, tau):
    hat_O = 1 / (1 + lambda_O * tau) * O
    hat_G = 1 / (1 + lambda_G * tau) * G + tau * lambda_G / (1 + lambda_G * tau) * np.ones_like(G)
    return hat_G, hat_O


def K(G, O, Y):
    shape_Y = Y.shape
    grad_hat_X = torch.zeros(shape_Y + (2,))
    for index_img in range(shape_Y[0]):
        grad_hat_X[index_img] = fwd_grad( G * Y[index_img] - O)
    return grad_hat_X

    
def trans_K(V, Y):
    shape_V = V.shape
    hat_G = 0
    hat_O = 0
    for index_img in range(shape_V[0]):
        div_img = - Div(V[index_img])
        hat_G += div_img * Y[index_img]
        hat_O += - div_img
    return np.array([hat_G, hat_O])


if __name__ == '__main__':
    
    G = torch.Tensor([[ i + 10*j for i in range(10)] for j in range(10)]).cuda()
    O = torch.Tensor([[ i*j +i**2 for i in range(10)] for j in range(10)]).cuda()
    
    grad_dl, grad_dr = fwd_dgrad(G)

    #G = torch.randint(0, 255, size=(3, 64, 64)).cuda()
    #O = torch.randint(0, 255, size=(64, 64, 3)).cuda()
    
    G_2 = torch.randint(0, 255, size=(64, 64, 3)).cuda()
    O_2 = torch.randint(0, 255, size=(64, 64, 3)).cuda()
    
    y = torch.Tensor([[ i*j +i**2 for i in range(64)] for j in range(64)]).cuda()
    
    Y = torch.randint(0, 255, size=(4, 64, 64, 3)).cuda()
    
    TV_loss = TV(y, False)
    
    #grad = fwd_grad(G, batch=True)
    #div = Div(grad)

    
    """
    trans_G_O = K(G, O, Y)
    G_O = torch.Tensor([G, O]) 
    trans_K_trans_G_O = trans_K(trans_G_O, Y)
    
    trans_G_O_2 = torch.Tensor([K(G_2, O_2, Y)])
    G_O_2 = torch.Tensor([G_2, O_2]) 
    trans_K_trans_G_O_2 = trans_K(trans_G_O_2, Y)
    
    print((G_O * trans_K_trans_G_O_2).sum() - (trans_G_O_2 * trans_G_O).sum())
    """
    
    