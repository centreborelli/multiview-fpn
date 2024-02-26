import numpy as np


def fwd_grad(img):
    img_shape = img.shape
    grad_j = np.diff(img, axis=1)
    grad_i = np.diff(img, axis=0)
    
    # if grayscale
    if len(img_shape)==2:
        grad_j = np.concatenate((grad_j, np.zeros((img_shape[0], 1))), axis=1)
        grad_i = np.concatenate((grad_i, np.zeros((1, img_shape[1]))), axis=0)
    else:
        grad_j = np.concatenate((grad_j, np.zeros((img_shape[0], 1, img_shape[2]))), axis=1)
        grad_i = np.concatenate((grad_i, np.zeros((1, img_shape[1], img_shape[2]))), axis=0)
    return grad_i, grad_j
    

def TV(img):
    grad_i, grad_j = fwd_grad(img)
    TV = np.sqrt(grad_i**2 + grad_j**2).sum()
    return TV


def Div(vec_f):
    vec_fi, vec_fj = vec_f
    vec_f_shape = vec_fi.shape
    
    vec_fj[:, -1] = 0
    vec_fi[-1, :] = 0
    
    if len(vec_f_shape)==2:
        vec_fj = np.concatenate((np.zeros((vec_f_shape[0], 1)), vec_fj), axis=1)
        vec_fi = np.concatenate((np.zeros((1, vec_f_shape[1])), vec_fi), axis=0)
    else:
        vec_fj = np.concatenate((np.zeros((vec_f_shape[0], 1, vec_f_shape[2])), vec_fj), axis=1)
        vec_fi = np.concatenate((np.zeros((1, vec_f_shape[1], vec_f_shape[2])), vec_fi), axis=0)
        
    div_vec_j = np.diff(vec_fj, axis=1)
    
    div_vec_i = np.diff(vec_fi, axis=0)
    
    return div_vec_i + div_vec_j


def proj_F_original(hat_V, M_y, tau):
    norm_tensor = np.sqrt((hat_V[:, 0, :] - tau * M_y[:, 0, :])**2 + (hat_V[:, 1, :] - tau * M_y[:, 1, :])**2)
    norm_tensor = np.maximum(1, norm_tensor)
    norm_tensor = np.repeat(norm_tensor[:, None, :], 2, axis=1)
    V = (hat_V - tau * M_y) / norm_tensor
    return V
    
    
def proj_J(O, lambda_O, tau):
    return 1 / (1 + lambda_O * tau) * O


def K(O, N):
    grad_O = np.array(fwd_grad(O))
    return np.repeat(grad_O[None], N, axis=0)

    
def trans_K(O):
    shape_O = O.shape
    hat_O = 0
    for index_img in range(shape_O[0]):
        hat_O += - Div(O[index_img])
    return hat_O


if __name__ == '__main__':
    
    O = np.array([[ i + 10*j for i in range(10)] for j in range(10)])
    
    O_2 = np.array([[ i*j +i**2 for i in range(10)] for j in range(10)])
    
    random_1 = np.random.randint(0, 255, size=(64, 64, 3))
    random_2 = np.random.randint(0, 255, size=(64, 64, 3))
    
    TV_loss = TV(random_1)
    print(TV_loss)
    
    grad = fwd_grad(random_1)
    div = Div(grad)
    
    img_K = K(random_1, 5)
    img_2_K = K(random_2, 5)
    print((random_1 * trans_K(img_2_K)).sum() - (K(random_1, 5) * img_2_K).sum())
    
    
    