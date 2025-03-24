import numpy as np
import cv2
from matplotlib import pyplot as plt
import gc
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import cg, spsolve, spilu, LinearOperator
import pyamg

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
        print(Ld.min(), Ld.max())
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
            beta = 0.8,
            maxiter = 10000):
        super().__init__(luminance_coefs, gamma)
        if beta == None:
            beta = 0.8
        self.beta = beta
        if maxiter == None:
            maxiter = 10000
        self.maxiter = maxiter

    def process(self, im):
        self.sigma_s = 0.02 * max(im.shape[0], im.shape[1])
        im_d = im.copy()

        Lw = self.compute_world_luminance(im)
        H = np.log((Lw) + 0.00001)
        gaussian_pyramid = self.compute_gaussian_pyramid(H)
        grads_pyramid = self.compute_gradient_pyramid(gaussian_pyramid)
        grad_H_x_k = grads_pyramid[0]
        grad_H_y_k = grads_pyramid[1]
        Phi = self.calculate_Phi(grad_H_x_k, grad_H_y_k)


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
        # print(Phi.min(), Phi.max())
        # cv2.imshow("Phi", Phi)
        # # cv2.imshow("Div G", div_G)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        

        # solve the poison equation
        row = []
        col = []
        val = []
        height = im.shape[0]
        width = im.shape[1]
        # poisson equation
        for x in range(height):
            for y in range(width):
                minus_cnt = 0
                # if x == 0 and y == 0:
                #     row.append(0)
                #     col.append(0)
                #     val.append(1.0)
                #     continue
                
                if x + 1 < height:
                    row.append(x * width + y)
                    col.append((x + 1) * width + y)
                    val.append(1.0)
                    minus_cnt += 1
                else:
                    row.append(x * width + y)
                    col.append((x) * width + y)
                    val.append(1.0)

                if x - 1 >= 0:
                    row.append(x * width + y)
                    col.append((x - 1) * width + y)
                    val.append(1.0)
                    minus_cnt += 1
                else:
                    row.append(x * width + y)
                    col.append((x) * width + y)
                    val.append(1.0)

                if y + 1 < width:
                    row.append(x * width + y)
                    col.append(x * width + (y + 1))
                    val.append(1.0)
                    minus_cnt += 1
                else:
                    row.append(x * width + y)
                    col.append(x * width + (y))
                    val.append(1.0)

                if y - 1 >= 0:
                    row.append(x * width + y)
                    col.append(x * width + (y - 1))
                    val.append(1.0)
                    minus_cnt += 1
                else:
                    row.append(x * width + y)
                    col.append(x * width + (y))
                    val.append(1.0)

                row.append(x * width + y)
                col.append(x * width + y)
                # val.append(-4)
                val.append(-1 * minus_cnt)


        A = csr_matrix((val, (row, col)), shape = (height * width, height * width))
        b = div_G.flatten()
        # print("start")
        # I_vec = spsolve(A, b)
        I_vec, exit_code = cg(A, b, maxiter=self.maxiter)
        # print(f"exit code: {exit_code}")
        print(f"residual: {np.sqrt((A * I_vec - b).dot((A * I_vec - b)))}")
        # ml = pyamg.ruge_stuben_solver(A)   
        # I_vec = ml.solve(b, tol=1e-10)
        # print(f"finish")
        Ld_log = I_vec.reshape((height, width))
        Ld = np.exp(Ld_log * self.gamma)
        # Ld -= Ld.min()
        black = 0.0005;
        white = 0.991;
        Ld_min = np.quantile(Ld, black)
        Ld_max = np.quantile(Ld, white)
        print(Ld.min(), Ld.max())
        print(Ld_min, Ld_max)
        Ld = np.clip(Ld, Ld_min, Ld_max)
        Ld = (Ld - Ld_min) / (Ld_max-Ld_min)
        # Ld = (Ld - Ld.min()) / ((Ld.max()) - Ld.min())
        # print(Ld_log.min(), Ld_log.max())
        # print(Ld.min(), Ld.max())
        # cv2.imshow("I log", (Ld_log - Ld_log.min()))# / (Ld_log.max()-Ld_log.min()))
        # # print(H.min(), H.max())
        # # cv2.imshow("H", (H - H.min())/(H.max() -H.min()))
        # cv2.imshow("I", (Ld - Ld.min()))#/(Ld.max() - Ld.min()))
        # # cv2.imshow("I", np.clip(Ld, 0.0001, Ld.max()))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        
        # convert luminance back to RGB
        Lw_3 = np.stack([Lw, Lw, Lw], axis=2)
        Ld_3 = np.stack([Ld, Ld, Ld], axis=2)
        im_d = Ld_3 * ((im / Lw_3))

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
        cur_H = cv2.filter2D(pyramid[-1], -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE)
        while height >= 32 and width >= 32:
            cur_H = cv2.resize(cur_H, (width,height), cv2.INTER_AREA)
            pyramid.append(cur_H)
            cur_H = cv2.filter2D(cur_H, -1, gaussian_kernel, borderType=cv2.BORDER_REPLICATE)
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
            #     print(H[0:3,0:3])
            #     print((cv2.filter2D(H, -1, gradx_kernel, borderType=cv2.BORDER_REPLICATE) / (2 ** (k + 1)))[0:3,0:3])
            #     print(k, (2 ** (k + 1)))
            gradx_pyramid.append(cv2.filter2D(H, -1, gradx_kernel, borderType=cv2.BORDER_REPLICATE) / (2 ** (k + 1)))
            grady_pyramid.append(cv2.filter2D(H, -1, grady_kernel, borderType=cv2.BORDER_REPLICATE) / (2 ** (k + 1)))

        return [gradx_pyramid, grady_pyramid]
    
    def calculate_Phi(self, grad_H_x, grad_H_y):
        # alpha = 0.1 * np.average(mag_grad_H_d)
        average = 0
        total_pixel_cnt = 0
        for k in range(len(grad_H_x)):
            average += np.sum(np.sqrt(grad_H_x[k] * grad_H_x[k] + grad_H_y[k] * grad_H_y[k]))
            total_pixel_cnt += grad_H_x[k].shape[0] * grad_H_x[k].shape[1]
        average /= total_pixel_cnt
        # alpha = 0.1 * np.average(np.sqrt(grad_H_x[0] * grad_H_x[0] + grad_H_y[0] * grad_H_y[0]))
        alpha = 0.1 * average
        mag_grad_H_d = np.sqrt(grad_H_x[-1] * grad_H_x[-1] + grad_H_y[-1] * grad_H_y[-1])
        
        cur_Phi = (((mag_grad_H_d + 1e-5) / alpha) ** (self.beta - 1))
        # cv2.imshow(f"Phi {len(grad_H_x) - 1}", cur_Phi)
        # print(f"d: {d}")
        for k in range(len(grad_H_x) - 2, -1, -1):
            mag_grad_H_k = np.sqrt(grad_H_x[k] * grad_H_x[k] + grad_H_y[k] * grad_H_y[k])
            # alpha = 0.1 * np.average(mag_grad_H_k)
            phi_k = (((mag_grad_H_k + 1e-5) / alpha) ** (self.beta - 1))
            height = mag_grad_H_k.shape[0]
            width = mag_grad_H_k.shape[1]
            L_Phi_k_plus_one = cv2.resize(cur_Phi, (width,height), cv2.INTER_LINEAR)
            cur_Phi = L_Phi_k_plus_one * phi_k
            # cv2.imshow(f"Phi {k} + 1", L_Phi_k_plus_one)
            # cv2.imshow(f"phi {k}", phi_k)
            # cv2.imshow(f"Phi {k}", cur_Phi)
            # print(f"cur Phi min max: {cur_Phi.min()} {cur_Phi.max()}")
            # print(f"cur Phi median: {np.median(cur_Phi)}")
            # print(cur_Phi.shape)
            # print(k)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        return cur_Phi