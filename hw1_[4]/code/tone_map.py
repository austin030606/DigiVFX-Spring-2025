import numpy as np
import cv2
from matplotlib import pyplot as plt
import gc
from scipy.sparse import csr_matrix 
from scipy.sparse.linalg import cg, spsolve

def gamma_correction(im, gamma):
    # suppose im is correct im_d hdr image
    im_gamma_corrected = ((im) ** (1 / gamma))
    return im_gamma_corrected


class ToneMap:
    def __init__(
            self,
            luminance_coefs = None,
            gamma = None):

        self.luminance_coefs = luminance_coefs
        self.gamma = gamma
        self.delta = 0.00001
        pass

    def process(self, im: np.ndarray):
        raise NotImplementedError()
    
    def compute_world_luminance(self, im):
        L = np.zeros((im.shape[0], im.shape[1]))
        B = im[:,:,0]
        G = im[:,:,1]
        R = im[:,:,2]
        # for i in range(im.shape[0]):
        #     for j in range(im.shape[1]):
        #         # apply the conversion for each pixel
        #         # for example if luminance_coefs = [0.06, 0.67, 0.27]
        #         # then L = 0.06B + 0.67G + 0.27R
        #         L[i][j] = self.luminance_coefs.dot(im[i][j])
        L = self.luminance_coefs[0] * B + self.luminance_coefs[1] * G + self.luminance_coefs[2] * R
        return L
    
    def get_log_average_luminance_of(self, L):
        log_sum = 0.0
        
        # for i in range(L.shape[0]):
        #     for j in range(L.shape[1]):
        #         log_sum += np.log(self.delta + L[i][j])
        L += self.delta
        log_sum = np.sum(np.log(L))

        return np.exp(log_sum / (L.shape[0] * L.shape[1]))

class ToneMapReinhard(ToneMap):
    def __init__(
            self, 
            luminance_coefs = np.array([0.06, 0.67, 0.27]), 
            gamma = None,
            delta = 0.00001, 
            a = 0.18, 
            L_white = None, 
            map_type = "global", 
            alphas = np.array([1.0/(2*(2**(1/2))), 1.6/(2*(2**(1/2)))]),
            scales = np.arange(1,43,2),
            phi = 8.0,
            epsilon = 0.05):
        
        super().__init__(luminance_coefs, gamma)
        
        self.delta = delta
        self.a = a
        self.L_white = L_white
        self.map_type = map_type
        self.alphas = alphas
        self.scales = scales
        self.phi = phi
        self.epsilon = epsilon

    def process(self, im):
        im_d = im.copy()

        Lw = self.compute_world_luminance(im)
        Lw_bar = self.get_log_average_luminance_of(Lw)
        
        L = (self.a / Lw_bar) * Lw

        Ld = None
        if self.map_type == "global":
            if self.L_white == None:
                self.L_white = np.max(L)

            # apply transformation for each pixel
            Ld = (L * (1 + (L / (self.L_white ** 2)))) / (1 + L)

        elif self.map_type == "local":
            R1 = self.compute_gaussian_kernels(1)
            R2 = self.compute_gaussian_kernels(2)

            # apply kernel and calculate V(x, y, s)
            V1 = []
            # V2 = []
            V = []
            for i, s in enumerate(self.scales):
                v1 = cv2.filter2D(L, -1, R1[i])
                v2 = cv2.filter2D(L, -1, R2[i])
                V1.append(v1)
                # V2.append(v2)
                V.append((v1 - v2)/((((2 ** self.phi) * self.a)/(s ** 2)) + v1))

            V1 = np.array(V1)
            # V2 = np.array(V2)
            V = np.array(V)
            # calculate s_max for each position
            s_m_idx = np.zeros((im.shape[0], im.shape[1]), np.uint)
            V1_s_m = np.zeros(V1[0].shape)
            for i in range(s_m_idx.shape[0]):
                for j in range(s_m_idx.shape[1]):
                    indices = np.where(np.abs(V[:,i,j]) > self.epsilon)[0]
                    if indices.size > 0:
                        idx = indices[0]
                        if idx > 0:
                            idx -= 1
                    else:
                        idx = self.scales.size - 1
                    # s_m_idx[i][j] = idx
                    V1_s_m[i][j] = V1[idx][i][j]
                    # for idx in range(self.scales.size):
                    #     if np.abs(V[idx][i][j]) < self.epsilon:
                    #         s_m_idx[i][j] = idx
                    #     else:
                    #         break
                print(f"{i}, {j}", end='\r')
            print("here")
            # apply transformation for each pixel
            Ld = np.zeros(L.shape)
            # for i in range(Ld.shape[0]):
            #     for j in range(Ld.shape[1]):
            #         Ld[i][j] = L[i][j] / (1 + V1[s_m_idx[i][j]][i][j])
            Ld = L / (1 + V1_s_m)
            print("done")

            del V
            del V1
            # del V2
            gc.collect()
            
        else:
            print("map type not implemented")
            raise NotImplementedError()

        # convert luminance back to RGB
        Lw_3 = np.stack([Lw, Lw, Lw], axis=2)
        Ld_3 = np.stack([Ld, Ld, Ld], axis=2)
        im_d = Ld_3 * (im / Lw_3)

        # print(np.min(im), np.max(im))
        # print(np.min(Lw), np.max(Lw))
        # print(np.min(Ld), np.max(Ld))
        # print(np.min(im_d), np.max(im_d))
        # apply gamma correction before returning if provided with gamma value
        if self.gamma == None:
            return im_d
        else:
            im_d_gamma_corrected = ((im_d) ** (1 / self.gamma))
            return im_d_gamma_corrected
    
    def compute_gaussian_kernels(self, alpha_i):
        kernels = []

        alpha = self.alphas[alpha_i - 1]
        for s in self.scales:
            kernel = np.zeros((s,s))
            for i in range(s):
                for j in range(s):
                    x = i - (s // 2)
                    y = j - (s // 2)
                    kernel[i][j] = (1 / (np.pi * ((alpha * s) ** 2))) * np.exp((-(x * x + y * y)) / ((alpha * s) ** 2))
            kernel /= np.sum(kernel)
            kernels.append(kernel)

        return kernels
    
class ToneMapDurand(ToneMap):
    def __init__(
            self, 
            luminance_coefs = np.array([1/61, 40/61, 20/61]),
            gamma = None,
            base_contrast = 4,
            limit_runtime = "Yes"):
        super().__init__(luminance_coefs, gamma)
        self.sigma_s = None
        self.sigma_r = 0.4
        if base_contrast == None:
            base_contrast = 4
        self.base_contrast = base_contrast
        if limit_runtime != "No":
            limit_runtime = True
        else:
            limit_runtime = False
        self.limit_runtime = limit_runtime

    def process(self, im):
        self.sigma_s = 0.02 * max(im.shape[0], im.shape[1])
        im_d = im.copy()

        Lw = self.compute_world_luminance(im)
        Lw_log = np.log(Lw + 0.00001).astype(np.float32)

        kernel_size = int(self.sigma_s * 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # f = self.compute_gaussian_kernel(self.sigma_s, kernel_size)
        # base = np.zeros(Lw_log.shape)
        # for i in range(base.shape[0]):
        #     for j in range(base.shape[1]):
        #         weighted_I = 0
        #         k = 0
        #         for x in range(kernel_size):
        #             for y in range(kernel_size):
        #                 di = x - (kernel_size // 2)
        #                 dj = y - (kernel_size // 2)
        #                 if i + di >= 0 and i + di < Lw_log.shape[0] and j + dj >= 0 and j + dj < Lw_log.shape[1]: 
        #                     weighted_I += f[x][y] * self.intensity_gaussian(np.abs(Lw_log[i][j] - Lw_log[i + di][j + dj])) * Lw_log[i + di][j + dj]
        #                     k += f[x][y] * self.intensity_gaussian(np.abs(Lw_log[i][j] - Lw_log[i + di][j + dj])) 
        #         base[i][j] = weighted_I / k
        #     print(f"row: {i}", end='\r')

        # the maximum sigma_s is set to this to limit the runtime when the input image is too large
        if self.limit_runtime and (kernel_size * max(im.shape[0], im.shape[1]) > 400000):
            self.sigma_s = 400000 / (max(im.shape[0], im.shape[1]) * 4)
            kernel_size = int(self.sigma_s * 4)
            # print(self.sigma_s)
            # print(kernel_size)
            if kernel_size % 2 == 0:
                kernel_size += 1

        base = cv2.bilateralFilter(Lw_log, kernel_size, self.sigma_r, self.sigma_s)

        detail = Lw_log - base
        scale = self.base_contrast / (np.max(base) - np.min(base))
        compressed = (base - np.max(base)) * scale
        Ld_log = compressed + detail
        print(Ld_log[0][0])
        Ld = np.exp(Ld_log)
        
        # convert luminance back to RGB
        Lw_3 = np.stack([Lw, Lw, Lw], axis=2)
        Ld_3 = np.stack([Ld, Ld, Ld], axis=2)
        im_d = Ld_3 * (im / Lw_3)

        # apply gamma correction before returning if provided with gamma value
        if self.gamma == None:
            return im_d
        else:
            im_d_gamma_corrected = ((im_d) ** (1 / self.gamma))
            return im_d_gamma_corrected
        
    def compute_gaussian_kernel(self, sigma, s):
        kernel = np.zeros((s,s))
        for i in range(s):
            for j in range(s):
                x = i - (s // 2)
                y = j - (s // 2)
                kernel[i][j] = np.exp((-(x * x + y * y)) / (2 * ((sigma) ** 2)))
        # kernel /= np.sum(kernel)
        return kernel
    
    def intensity_gaussian(self, d):
        return np.exp((-(d * d)) / (2 * ((self.sigma_r) ** 2)))
    
class ToneMapFattal(ToneMap):
    def __init__(
            self, 
            luminance_coefs = np.array([1/61, 40/61, 20/61]),
            gamma = None,
            beta = 0.8):
        super().__init__(luminance_coefs, gamma)
        if beta == None:
            beta = 0.8
        self.beta = beta

    def process(self, im):
        self.sigma_s = 0.02 * max(im.shape[0], im.shape[1])
        im_d = im.copy()

        Lw = self.compute_world_luminance(im)
        H = np.log(Lw + 0.00001).astype(np.float32)
        gaussian_pyramid = self.compute_gaussian_pyramid(H)
        grad_H_x_k, grad_H_y_k = self.compute_gradient_pyramid(gaussian_pyramid)
        Phi = self.calculate_Phi(grad_H_x_k, grad_H_y_k)
        # exit()

        grady_kernel = np.array([[0,0,0],
                                 [0,-1,1],
                                 [0,0,0]])
        gradx_kernel = np.array([[0,0,0],
                                 [0,-1,0],
                                 [0,1,0]])
        divy_kernel = np.array([[0,0,0],
                                [-1,1,0],
                                [0,0,0]])
        divx_kernel = np.array([[0,-1,0],
                                [0,1,0],
                                [0,0,0]])
        grad_H_x = cv2.filter2D(H, -1, gradx_kernel, borderType=cv2.BORDER_REPLICATE)
        grad_H_y = cv2.filter2D(H, -1, grady_kernel, borderType=cv2.BORDER_REPLICATE)
        height = im.shape[0]
        width = im.shape[1]
        G_x = grad_H_x * Phi
        G_y = grad_H_y * Phi
        div_G = cv2.filter2D(G_x, -1, divx_kernel, borderType=cv2.BORDER_REPLICATE) + cv2.filter2D(G_y, -1, divy_kernel, borderType=cv2.BORDER_REPLICATE)
        # print(cv2.filter2D(G_x, -1, divx_kernel, borderType=cv2.BORDER_REPLICATE)[0])
        # print(grad_H_x[-1])
        cv2.imshow("I", ((Phi - Phi.min()) / (Phi.max() - Phi.min())))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
        

        # solve the poison equation
        row = []
        col = []
        val = []
        cnt = 0
        b = []
        height = im.shape[0]
        width = im.shape[1]
        # poisson equation
        for x in range(height):
            for y in range(width):
                if x == 0 and y == 0 and False:
                    row.append(cnt)
                    col.append(x * width + y)
                    val.append(1.0)

                    b.append(-4.9115515)

                    cnt += 1
                else:
                    if x + 1 < height:
                        row.append(cnt)
                        col.append((x + 1) * width + y)
                        val.append(1.0)
                    else:
                        # Reflect at right boundary
                        row.append(cnt)
                        col.append(x * width + y)
                        val.append(1.0)

                    if x - 1 >= 0:
                        row.append(cnt)
                        col.append((x - 1) * width + y)
                        val.append(1.0)
                    else:
                        row.append(cnt)
                        col.append(x * width + y)
                        val.append(1.0)

                    if y + 1 < width:
                        row.append(cnt)
                        col.append(x * width + (y + 1))
                        val.append(1.0)
                    else:
                        row.append(cnt)
                        col.append(x * width + (y))
                        val.append(1.0)

                    if y - 1 >= 0:
                        row.append(cnt)
                        col.append(x * width + (y - 1))
                        val.append(1.0)
                    else:
                        row.append(cnt)
                        col.append(x * width + (y))
                        val.append(1.0)

                    row.append(cnt)
                    col.append(x * width + y)
                    val.append(-4.0)

                    b.append(div_G[x][y])

                    cnt += 1


        A = csr_matrix((val, (row, col)), shape = (cnt, height * width)) 
        b = np.array(b)
        I_vec = spsolve(A, b)
        # I_vec, exit_code = cg(A, b)
        # print(f"exit code: {exit_code}")
        Ld_log = np.zeros((im.shape[0], im.shape[1]))
        for x in range(height):
            for y in range(width):
            #    x = i + 1
            #    y = j + 1
               Ld_log[x][y] = I_vec[x * width + y]
        Ld_log = ((Ld_log - Ld_log.min()) / (Ld_log.max() - Ld_log.min()))
        print(Ld_log.min(), Ld_log.max())
        cv2.imshow("I", ((Ld_log - Ld_log.min()) / (Ld_log.max() - Ld_log.min())))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
        print(Ld_log.min(), Ld_log.max())
        Ld = np.exp(Ld_log)
        print(Ld.min(), Ld.max())
        
        # convert luminance back to RGB
        Lw_3 = np.stack([Lw, Lw, Lw], axis=2)
        Ld_3 = np.stack([Ld, Ld, Ld], axis=2)
        im_d = Ld_3 * (im / Lw_3)

        # apply gamma correction before returning if provided with gamma value
        if self.gamma == None:
            return im_d
        else:
            im_d_gamma_corrected = ((im_d) ** (1 / self.gamma))
            return im_d_gamma_corrected
        
    def compute_gaussian_pyramid(self, H):
        gaussian_kernel = (1/16) * np.array([[1,2,1],
                                             [2,4,2],
                                             [1,2,1]])
        pyramid = [H.copy()]
        height = H.shape[0] // 2
        width = H.shape[1] // 2
        while height >= 32 and width >= 32:
            cur_H = cv2.resize(H, (width,height), cv2.INTER_AREA)
            cur_H = cv2.filter2D(cur_H, -1, gaussian_kernel)
            pyramid.append(cur_H)
            height //= 2
            width //= 2

        return pyramid
    
    def compute_gradient_pyramid(self, pyramid):
        grady_kernel = np.array([[0,0,0],
                                 [-1,0,1],
                                 [0,0,0]])
        gradx_kernel = np.array([[0,-1,0],
                                 [0,0,0],
                                 [0,1,0]])
        
        gradx_pyramid = []
        grady_pyramid = []
        for k, H in enumerate(pyramid):
            # if k == len(pyramid) - 1:
            #     print(cv2.filter2D(H, -1, gradx_kernel / (2 ** (k + 1)), borderType=cv2.BORDER_REPLICATE)[0:3,0:3])
            #     print(H[0:3,0:3])
            #     print(k)
            gradx_pyramid.append(cv2.filter2D(H, -1, gradx_kernel / (2 ** (k + 1)), borderType=cv2.BORDER_REPLICATE))
            grady_pyramid.append(cv2.filter2D(H, -1, grady_kernel / (2 ** (k + 1)), borderType=cv2.BORDER_REPLICATE))

        return gradx_pyramid, grady_pyramid
    
    def calculate_Phi(self, grad_H_x, grad_H_y):
        d = len(grad_H_y) - 1
        mag_grad_H_d = np.sqrt(grad_H_x[d] * grad_H_x[d] + grad_H_y[d] * grad_H_y[d])
        alpha = 0.1 * np.average(mag_grad_H_d)
        
        print("**invalid case handled when using np.where, so it is safe to ignore the following warnings regarding division by zero and invalid value**")
        cur_Phi = np.where(
            mag_grad_H_d == 0,
            0,
            (alpha / mag_grad_H_d) * ((mag_grad_H_d / alpha) ** self.beta)
        )
        # print(f"d: {d}")
        for k in range(d - 1, -1, -1):
            mag_grad_H_k = np.sqrt(grad_H_x[k] * grad_H_x[k] + grad_H_y[k] * grad_H_y[k])
            alpha = 0.1 * np.average(mag_grad_H_k)
            # print("**invalid case handled when using np.where, so it is safe to ignore the two following warnings**")
            phi_k = np.where(
                mag_grad_H_k == 0,
                0,
                (alpha / mag_grad_H_k) * ((mag_grad_H_k / alpha) ** self.beta)
            )
            height = mag_grad_H_k.shape[0]
            width = mag_grad_H_k.shape[1]
            L_Phi_k_plus_one = cv2.resize(cur_Phi, (width,height), cv2.INTER_LINEAR)
            cur_Phi = L_Phi_k_plus_one * phi_k
            # print(cur_Phi.shape)
            # print(k)
        
        return cur_Phi