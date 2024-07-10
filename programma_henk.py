import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Model():
    def __init__(self):
        # Configurable constants
        self.N = 50
        self.a = 1 #  nm
        self.rho = np.array([2.330] * (self.N+1)) # g/cm^3
        self.EV = np.array([0]*(self.N+1)) # nm
        self.l_b = np.array([5e9] * (self.N+1)) # 1/s
        self.k_t = np.array([0] * (self.N+1))  # 
        self.n_t = np.array([1e-4] * (self.N+1))  #
        self.L_a = 1e-10 # nm

        self.Al_index = np.arange(19)
        self.AlSio_index = 19
        self.Sio_index = np.arange(20,23)
        self.Siosi1 = 23
        self.Siosi2 = 24
        self.Si_index = np.arange(25,self.N)

        self.S_i = np.zeros(self.N)
        self.S_i[self.Al_index] = 0.61
        self.S_i[self.AlSio_index] = 0.56
        self.S_i[self.Sio_index] = 0.56
        self.S_i[self.Siosi1] = 0.58
        self.S_i[self.Siosi2] = 0.58
        self.S_i[self.Si_index] = 0.576
        self.S_surf = 0.57

        self.W_i = np.zeros(self.N)
        self.W_i[self.Al_index] = 0.025
        self.W_i[self.AlSio_index] = 0.048
        self.W_i[self.Sio_index] = 0.048
        self.W_i[self.Siosi1] = 0.04
        self.W_i[self.Siosi2] = 0.04
        self.W_i[self.Si_index] = 0.027
        self.W_surf = 0.032

        self.F_surf = 1

        self.L_p = np.zeros(self.N+1) # nm
        self.L_p[self.Al_index] = 10
        self.L_p[self.AlSio_index] = 10
        self.L_p[self.Sio_index] = 10
        self.L_p[self.Siosi1] = 10
        self.L_p[self.Siosi2] = 10
        self.L_p[self.Si_index] = 100
        self.L_p[-1] = 100

        self.update_val()

    def update_val(self):
        self.v_d =  self.a*self.EV/self.L_p**2 #

        self.alpha = 1 / self.L_p**2

        dz = 0.6 #nm
        inc = 1.3
        self.z = [0]
        for i in range(0,self.N):
            self.z.append(dz * inc**i)

        self.z[18] = 45
        self.z[19] = 50
        self.z[22] = 150
        self.z[23] = 155
        self.z[24] = 160

        self.z = np.array(self.z)/self.a # z -> z*

        # plt.figure()
        # plt.plot(self.z[:28])
        # plt.show()
        # for i in range(len(self.z)):
        #     print(i, self.z[i])
        # quit()

        dE = 0.2 #keV
        inc = 1.1
        self.E_space = [.1]
        for i in range(0,52):
            self.E_space.append(dE * inc**i)



        # self.z=np.linspace(0,1500,self.N+1)/self.a

    # Definitions
    def gamma(self, sign, i):
        return 0.5*(self.v_d[i] + sign* self.a* ( ( self.v_d[i]/self.a)**2 + 4/self.L_p[i]**2 )**.5)

    def q(self, i):
        arg = -self.gamma(1,i)*(self.z[i-1]-self.z[i])
        # print(arg)
        if arg < 50:
            return 1/(np.exp(arg)-1)
        else:
            return np.exp(-arg)

    def r(self, i):
        arg = self.gamma(1,i)*(self.z[i-1]-self.z[i])
        return 1/(1-np.exp(arg))

    def t(self,i):
        arg = self.gamma(-1,i)*(self.z[i-1]-self.z[i])

        if arg < 50:
            return 1/(1-np.exp(arg))
        else:
            return -np.exp(-arg)

    def s(self, i):
        arg = -self.gamma(-1,i)*(self.z[i-1]-self.z[i])

        return 1/(np.exp(arg)-1)

    def alph(self, sign,i):
        return self.r(i) + sign* self.t(i)

    def beta(self, sign,i):
        return self.q(i) + sign* self.s(i)

    def delta(self, sign, i):
        return self.r(i)/self.gamma(-1,i) + sign*self.t(i)/self.gamma(1,i)

    def eps(self, sign, i):
        return self.q(i)/self.gamma(-1,i) + sign*self.s(i)/self.gamma(1,i)

    def Delta(self, i):
        return self.v_d[i] - self.v_d[i - 1]


    def insert_interval(self, i,j):
        if i==j:
            if i == 0:
                return 1/self.z[0]
            return 1/(self.z[i]-self.z[i-1])
        
        return 0
    
    # Calculations matrices
    def C(self, j):
        A = np.zeros((self.N + 1, self.N + 1))
        b = np.zeros((self.N + 1))

        A[0, 0] = self.eps(1,1)/self.eps(-1,1)*(self.alph(-1,1)-self.a/self.L_a*self.delta(-1,1)) - self.alph(1,1)+self.a/self.L_a*self.delta(1,1)
        A[0, 1] = self.beta(1,1)-self.eps(1,1)/self.eps(-1,1)*self.beta(-1,1)

        b[0] = -2*self.insert_interval(1,j)/self.alpha[0]/self.a**2

        def a2(i):
            return -self.eps(-1,i+1)/self.eps(1,i+1)
        def a3(i):
            return (self.delta(-1,i+1)+self.delta(1,i+1)*a2(i))/(self.eps(-1,i)-self.delta(-1,i)/self.delta(1,i)*self.eps(1,i))
        def a1(i):
            return -self.delta(-1,i)/self.delta(1,i)*a3(i)

        for i in range(1, self.N):
            A[i, i - 1] = self.alph(-1,i)*a3(i) + self.alph(1,i)*a1(i)
            A[i, i] = a3(i)*(self.eps(-1,i)*self.Delta(i)-self.beta(-1,i)) + a1(i)*(self.eps(1,i)*self.Delta(i) - self.beta(1,i)) + self.alph(-1,i+1) + a2(i)*self.alph(1,i+1)
            A[i, i + 1] = self.eps(-1,i+1)*self.Delta(i+1)-self.beta(-1,i+1) + a2(i)*(self.eps(1,i+1)*self.Delta(i+1)-self.beta(1,i+1))
            b[i] = (2* a1(i)* self.insert_interval(i,j)/self.alpha[i] + 2* a2(i)* self.insert_interval(i+1,j)/self.alpha[i+1] )/self.a**2
            
        A[self.N, self.N - 1] = -self.alph(1,self.N) + self.delta(1,self.N)/self.delta(-1,self.N)*self.alph(-1,self.N)
        A[self.N, self.N] = self.beta(1,self.N) + self.eps(1,self.N)*self.a/self.L_p[self.N] - self.delta(1,self.N)/self.delta(-1,self.N)*(self.beta(-1,self.N)+self.eps(-1,self.N)*self.a/self.L_p[self.N])

        b[self.N] = -2*self.insert_interval(self.N,j)/self.alpha[self.N]/self.a**2

        C = np.matmul(np.linalg.inv(A), b)#*self.a**2/(self.L_p**2*self.l_b)

        return C

    def C_deriv(self, j):
        A = np.zeros((self.N + 1, self.N + 1))
        b = np.zeros((self.N + 1))

        f = (self.beta(1,1)-self.eps(1,1)*self.Delta(1))/(self.beta(-1,1)-self.eps(-1,1)*self.Delta(1))

        A[0, 0] = f*(self.alph(-1,1)*self.L_a/self.a - self.delta(-1,1)) - self.alph(1,1)*self.L_a/self.a + self.delta(1,1)
        A[0, 1] = f*self.eps(-1,1) - self.eps(1,1)

        b[0] = -2*self.insert_interval(1,j)/self.alpha[0]/self.a**2

        def a2(i):
            return -(self.beta(-1,i+1)-self.eps(-1,i+1)*self.Delta(i+1))/(self.beta(1,i+1)-self.eps(1,i+1)*self.Delta(i+1))
        def a3(i):
            return -(self.alph(-1,i+1)+self.alph(1,i+1)*a2(i))/(self.eps(-1,i)*self.Delta(i)-self.beta(-1,i) - self.alph(-1,i)/self.alph(1,i)*(self.eps(1,i)*self.Delta(i)-self.beta(1,i)))
        def a1(i):
            return -self.alph(-1,i)/self.alph(1,i)*a3(i)

        for i in range(1, self.N):
            A[i, i - 1] = -a1(i)*self.delta(1,i) - a3(i)*self.delta(-1,i)
            A[i, i] = a3(i)*self.eps(-1,i) - self.delta(-1,i+1) + a1(i)*self.eps(1,i) - a2(i)*self.delta(1,i+1)
            A[i, i + 1] = self.eps(-1,i+1)+a2(i)*self.eps(1,i+1)
            b[i] = (2* a1(i)* self.insert_interval(i,j)/self.alpha[i] + 2* a2(i)* self.insert_interval(i+1,j)/self.alpha[i+1] )/self.a**2
            
        A[self.N, self.N - 1] = self.delta(1,self.N) - self.alph(1,self.N)/self.alph(-1,self.N)*self.delta(-1,self.N)
        A[self.N, self.N] = self.alph(1,self.N)/self.alph(-1,self.N)*(self.beta(-1,self.N)*self.L_p[self.N]/self.a+self.eps(-1,self.N)) - self.beta(1,self.N)*self.L_p[self.N]/self.a - self.eps(1,self.N)
        b[self.N] = -2*self.insert_interval(self.N,j)/self.alpha[self.N]/self.a**2

        C = np.matmul(np.linalg.inv(A), b)#*self.a**2/(self.L_p**2*self.l_b)

        return C

    # Determine S and F fractions
    def P_j(self, E, j):
        m = 2
        n = 1.62
        alpha = 40

        z_0 = 1.13* alpha / self.rho[j] * E**n

        if j == 0:
            return 1
        return np.exp(-(self.z[j-1]/z_0)**m) - np.exp(-(self.z[j]/z_0)**m)


    def T_ij(self, progress_callback=None):
        T_ij = np.zeros((self.N,self.N))

        # j for implantation in interval j
        for j in range(1, self.N):
            if progress_callback:
                progress_callback.emit(j)
            else:
                print(j)


            C_res = self.C(j)
            C_der_res = self.C_deriv(j)

            # Calc for each interval i
            for i in range(1,self.N+1):
                T_ij[i-1,j-1] = C_der_res[i]-C_der_res[i-1] + (i==j) - self.v_d[i]*C_res[i] + self.v_d[i-1]*C_res[i-1]

        self.T_ij_calc = T_ij
        return T_ij

    def T_i(self, T_ij_calc, E):
        T_i = np.zeros(self.N)

        for j in range(self.N):
            T_i += T_ij_calc[:,j]*self.P_j(E,j)

        return T_i

    def T_surf(self, E):
        return 1-np.sum(self.T_i(self.T_ij_calc, E))

    def S(self, E):
        return np.sum(self.S_i*self.T_i(self.T_ij_calc, E)) + self.T_surf(E)*self.S_surf
    
    def W(self, E):
        return np.sum(self.W_i*self.T_i(self.T_ij_calc, E)) + self.T_surf(E)*self.W_surf
    
    def S_fit(self, E, S_al, S_alsio, S_sio, S_siosi, S_si, S_surf):
        self.S_i = np.zeros(self.N)
        self.S_i[self.Al_index] = S_al
        self.S_i[self.AlSio_index] = S_alsio
        self.S_i[self.Sio_index] = S_sio
        self.S_i[self.Siosi] = S_siosi
        self.S_i[self.Si_index] = S_si
        self.S_surf = S_surf
        

        return [self.S(e) for e in E]
    
    def W_fit(self, E, W_al, W_alsio, W_sio, W_siosi, W_si, W_surf):
        self.W_i = np.zeros(self.N)
        self.W_i[self.Al_index] = W_al
        self.W_i[self.AlSio_index] = W_alsio
        self.W_i[self.Sio_index] = W_sio
        self.W_i[self.Siosi] = W_siosi
        self.W_i[self.Si_index] = W_si
        self.W_surf = W_surf
        

        return [self.S(e) for e in E]


if __name__ == "__main__":
    model = Model()
    model.T_ij()
    # print(model.T_ij_calc[0])

    # model2 = Model()
    # model2.EV = np.zeros(model2.N+1)
    # model2.EV[18:23] = 30
    # # print(model2.EV)
    # model2.update_val()
    # model2.T_ij()

    # model3 = Model()
    # model3.EV = np.array([-300]*(model3.N+1))
    # model3.update_val()
    # model3.T_ij()

    # print(np.trapz(model.C_deriv(48), model.z*model.a))
    # print(model.z[25]*model.a)

    plt.figure()
    # plt.plot(0.0075*(model.C(14)[:-1]-model.C(14)[1:]))
    # plt.plot(model.C_deriv(14)[1:]-model.C_deriv(14)[:-1])
    plt.plot(model.z*model.a, model.C_deriv(25))
    # plt.plot(model.z*model.a, model.C(45))
    # plt.plot(model.z*model.a,model.C_deriv(25))
    # plt.plot(np.arange(1,31), model.T_ij_calc[0], label="EV = 0")
    # plt.plot(np.arange(1,31),model2.T_ij_calc[0], label="EV = 300")
    # plt.plot(np.arange(1,31),model3.T_ij_calc[0], label="EV = -300")
    # for i in range(model.N):
    #     plt.plot(np.arange(1,31),model.T_ij_calc[i])
    
    # plt.plot(np.arange(1,31), [np.trapz(model.T_ij_calc[i]*model.insert_interval(i,i), model.z[:-1]*model.a) for i in range(model.N)])
    # plt.plot(np.arange(1,31), [np.max(model.T_ij_calc[i]) for i in range(model.N)])
    # plt.plot(model.z*model.a,model.C_deriv(40))
    # plt.plot(model.z*model.a,model.C_deriv(45))
    # plt.plot([model.P_j(20,j) for j in np.arange(31)])
    # plt.plot(model.z[:-1]*model.a, model.T_i(model.T_ij_calc,20), label=f"{np.sum(model.T_i(model.T_ij_calc,20))}, {np.trapz(model.T_i(model.T_ij_calc,20),model.z[:-1]*model.a)}")
    # plt.plot(model.z[:-1]*model.a, model.T_i(model.T_ij_calc,10), label=f"{np.sum(model.T_i(model.T_ij_calc,10))}, {np.trapz(model.T_i(model.T_ij_calc,10),model.z[:-1]*model.a)}")
    plt.xlabel("z (nm)")
    plt.ylabel("C'*")
    plt.xlim(0,2000)
    # plt.legend()
    plt.xlim(left=0)
    # plt.ylim(bottom=0)
    plt.grid()
    plt.show()

    # print(np.sum([model.P_j(20,j) for j in np.arange(31)]))
    exp_data = np.loadtxt("../exp data.csv", delimiter=",", skiprows=1)
    print(exp_data)
    E_data = exp_data[:,0]
    W_data = exp_data[:,1]
    S_data = exp_data[:,2]
    # print(E_data, W_data)

    # print()
    # E_data = np.fromstring(E_data.replace('\n',','),sep=",")
    # W_data = np.fromstring(W_data.replace('\n',','),sep=",")
    # S_data = np.fromstring(S_data.replace('\n',','),sep=",")

    E = np.linspace(.1,25,1000)

    # poptS,pcovS = curve_fit(model.S_fit, E_data, S_data, bounds=(0.3,0.7), p0=[0.61,0.556,0.5499,0.576,0.57])
    # poptW,pcovW = curve_fit(model.W_fit, E_data, W_data, bounds=(0,0.1), p0=[0.025, 0.054, 0.027, 0.027, 0.032])

    # print(poptS, poptW)
    # model.S_fit(E_data, *poptS)
    # model.W_fit(E_data, *poptW)

    # print(model.T_ij_calc[25])

    # # np.savetxt("int.csv",T_ij_calc)
    # # print(T_ij_calc)
    # # print(T_i, S(np.array([0.5]*N), 0, 20))

    # # print(np.sum(T(20)))
    # model.T_i(model.T_ij_calc, 20)

    plt.figure()
    # plt.plot(E, [np.sum([model.P_j(e,j) for j in np.arange(model.N+1)]) for e in E])
    # plt.plot(E, [model.T_surf(e) for e in E])
    # plt.plot(E_data, S_data)
    # plt.plot(E_data, [model.S(e) for e in E_data])
    plt.plot([model.S(e) for e in E_data], [model.W(e) for e in E_data], label="EV=0")
    # plt.plot([model2.S(e) for e in E], [model2.W(e) for e in E], label="EV=3")
    plt.plot(S_data, W_data, label="Experiment 0V")
    # plt.plot(E, [model2.S(e) for e in E], label="EV=300")
    # plt.plot(E, [model3.S(e) for e in E], label="EV=-300")
    # plt.plot(E, [model.F(e) for e in E])
    # plt.plot([model.F(e) for e in E], [model.S(e) for e in E])
    plt.xlabel("S")
    plt.ylabel("W")
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.plot(E_data, W_data)
    # plt.plot(E_data, [model.W(e) for e in E_data])
    # plt.show()
