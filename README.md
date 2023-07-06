# Domain evolution in ferroelectric materials
## Free energy density

$$F=\alpha_{ij}P_iP_j+\alpha_{ijkl}P_iP_jP_kP_l+\alpha_{ijklmn}P_iP_jP_kP_lP_mP_n+\alpha_{ijklmnor}P_iP_jP_kP_lP_mP_nP_oP_r$$

$$+\frac{1}{2}g_{ijkl}P_{i,j}P_{k,l}+\frac{1}{2}c_{ijkl}\varepsilon_{ij}\varepsilon_{kl}-q_{ijkl}\varepsilon_{ij}P_kP_l+\frac{1}{2}f_{ijkl}(\varepsilon_{ij}P_{k,l}-\varepsilon_{ij,l}P_k)-\frac{1}{2}\epsilon_0E_iE_i-P_iE_i$$

$11\to 1;22\to 2;12\to 3;$

$$F=\alpha_1(P^2_1+P^2_2)+\alpha_{11}(P^4_1+P^4_2)+\alpha_{12}P^2_1P^2_2+\alpha_{111}(P^6_1+P^6_2)+\alpha_{112}(P^2_1P^4_2+P^4_1P^2_2)+\alpha_{1111}(P^8_1+P^8_2)+\alpha_{1122}P^4_1P^4_2+\alpha_{1112}(P^2_1P^6_2+P^6_1P^2_2)$$

$$+\frac{1}{2}g_{11}(P^2_{1,1}+P^2_{2,2})+g_{12}P_{1,1}P_{2,2}+\frac{1}{2}g_{33}(P_{1,2}+P_{2,1})^2+\frac{1}{2}c_{11}(\varepsilon^2_1+\varepsilon^2_2)+c_{12}\varepsilon_1\varepsilon_2+\frac{1}{2}c_{33}\varepsilon^2_3-q_{11}(\varepsilon_1P^2_1+\varepsilon_2P^2_2)$$

$$-q_{12}(\varepsilon_1P^2_2+\varepsilon_2P^2_1)-q_{33}\varepsilon_3P_1P_2+\frac{1}{2}f_{11}(\varepsilon_1P_{1,1}-\varepsilon_{1,1}P_1)+\frac{1}{2}f_{22}(\varepsilon_2P_{2,2}-\varepsilon_{2,2}P_2)-\frac{1}{2}\epsilon_0(E_1E_1+E_2E_2)-P_1E_1-P_2E_2$$

## Constitutive relation

$$D_i=-\frac{\partial F}{\partial E_i}=\epsilon_0E_i+P_i$$

## From strain gradient elasticity

$$\kappa_{kij}=\varepsilon_{ij,k}$$

### hyperstress
$$\tau_{lij}=\frac{\partial F}{\partial \kappa_{kij}}=\frac{\partial F}{\partial \varepsilon_{ij,k}}=-\frac{1}{2}f_{ijkl}P_k$$

$$\tau_{lij,lj}=-\frac{1}{2}f_{ijkl}P_{k,lj}$$

### stress
$$\sigma_{ij}=\frac{\partial F}{\partial \varepsilon_{ij}}=c_{ijkl}\varepsilon_{kl}-q_{ijkl}P_kP_l+\frac{1}{2}f_{ijkl}P_{k,l}$$

### strain in small deformation and electric field
$$\varepsilon_{ij}=\frac{1}{2}(u_{i,j}+u_{j,i})$$
$$E_i=-\varphi_{,i}$$

## Mechanical balance equation

$$\sigma_{ij,j}-\tau_{lij,lj}=0$$
$$c_{ijkl}\varepsilon_{kl,j}-q_{ijkl}(P_kP_{l,j}+P_{k,j}P_l)+f_{ijkl}P_{k,lj}=0$$
$$\begin{cases}
\sigma_{11,1}+\sigma_{12,2}-\tau_{111,11}-\tau_{212,22}-\tau_{112,12}-\tau_{211,21}=0(1)\\
\sigma_{21,1}+\sigma_{22,2}-\tau_{121,11}-\tau_{221,21}-\tau_{222,22}-\tau_{122,12}=0(2)
\end{cases}$$

$$\begin{cases}
c_{11}u_{1,11}+c_{31}u_{1,12}+c_{13}(u_{1,21}+u_{2,11})+c_{33}(u_{1,22}+u_{2,12})+c_{12}u_{2,21}+c_{32}u_{2,22}-2q_{11}P_1P_{1,1}-2q_{31}P_1P_{1,2}-2q_{13}(P_2P_{1,1}+P_1P_{2,1})\\
-2q_{33}(P_1P_{2,2}+P_2P_{1,2})-2q_{12}P_2P_{2,1}-2q_{32}P_2P_{2,2}+f_{11}P_{1,11}+f_{31}P_{1,12}+f_{13}(P_{2,11}+P_{1,21})+f_{33}(P_{2,12}+P_{1,22})+f_{12}P_{2,12}+f_{32}P_{2,22}=0(1)\\
c_{31}u_{1,11}+c_{21}u_{1,12}+c_{33}(u_{1,21}+u_{2,11})+c_{23}(u_{1,22}+u_{2,12})+c_{32}u_{2,21}+c_{22}u_{2,22}-2q_{31}P_1P_{1,1}-2q_{21}P_1P_{1,2}-2q_{33}(P_1P_{2,1}+P_2P_{1,1})\\
-2q_{23}(P_1P_{2,2}+P_2P_{1,2})-2q_{32}P_2P_{2,1}-2q_{22}P_2P_{2,2}+f_{31}P_{1,11}+f_{21}P_{1,21}+f_{33}(P_{2,11}+P_{1,21})+f_{23}(P_{2,21}+P_{1,22})+f_{32}P_{2,21}+f_{22}P_{2,22}=0(2)
\end{cases}$$


## Maxwell equation

$$D_{i,i}=0$$

$$D_{1,1}+D_{2,2}=0$$

$$\epsilon_0E_{1,1}+\epsilon_0E_{2,2}+P_{1,1}+P_{2,2}=0$$

$$-\epsilon_0\varphi_{,11}-\epsilon_0\varphi_{,22}+P_{1,1}+P_{2,2}=0(3)$$

## Phase-field AC equation

$$\frac{1}{L}\dot P_k-\frac{\partial F_L}{\partial P_k}+g_{ijkl}P_{i,jl}+2q_{ijkl}\varepsilon_{ij}P_l+f_{ijkl}\varepsilon_{ij,l}+E_k=0$$

$$\begin{cases}
\frac{1}{L}\dot P_1-\frac{\partial F_L}{\partial P_1}+g_{ij1l}P_{i,jl}+2q_{ij1l}\varepsilon_{ij}P_l+f_{ij1l}\varepsilon_{ij,l}+E_1=0(4)\\
\frac{1}{L}\dot P_2-\frac{\partial F_L}{\partial P_2}+g_{ij1l}P_{i,jl}+2q_{ij1l}\varepsilon_{ij}P_l+f_{ij1l}\varepsilon_{ij,l}+E_2=0(5)
\end{cases}$$

$$F_L=\alpha_1(P^2_1+P^2_2)+\alpha_{11}(P^4_1+P^4_2)+\alpha_{12}P^2_1P^2_2+\alpha_{111}(P^6_1+P^6_2)+\alpha_{112}(P^2_1P^4_2+P^4_1P^2_2)+\alpha_{1111}(P^8_1+P^8_2)+\alpha_{1122}P^4_1P^4_2+\alpha_{1112}(P^2_1P^6_2+P^6_1P^2_2)$$

$$\begin{cases}
\dot P_1-(2\alpha_1P_1+4\alpha_{11}P^3_1+2\alpha_{12}P_1P^2_2+6\alpha_{111}P^5_1+\alpha_{112}(2P_1P^4_2+4P^3_1P^2_2)+8\alpha_{1111}P^7_1+4\alpha_{1122}P^3_1P^4_2\\
+\alpha_{1112}(2P_1P^6_2+6P^5_1P^2_2)+g_{11}P_{1,11}+g_{31}P_{2,11}+g_{31}P_{1,21}+g_{13}P_{1,12}+g_{21}P_{2,21}+g_{33}P_{2,12}+g_{33}P_{1,22}+g_{23}P_{2,22}\\
+2q_{11}u_{1,1}P_1+2q_{31}(u_{1,2}+u_{2,1})P_1+2q_{13}u_{1,1}P_2+2q_{21}u_{2,2}P_1+2q_{33}(u_{1,2}+u_{2,1})P_2+2q_{23}u_{2,2}P_2+f_{11}u_{1,11}\\
+f_{31}(u_{1,21}+u_{2,11})+f_{13}u_{1,12}+f_{21}u_{2,21}+f_{33}(u_{1,22}+u_{2,12})+f_{23}u_{2,22}-\varphi_{,1}=0(4)\\
\dot P_2-(2\alpha_1P_2+4\alpha_{11}P^3_2+2\alpha_{12}P_2P^2_1+6\alpha_{111}P^5_2+\alpha_{112}(2P_2P^4_1+4P^3_2P^2_1)+8\alpha_{1111}P^7_2+4\alpha_{1122}P^3_2P^4_1\\
+\alpha_{1112}(2P_2P^6_1+6P^5_2P^2_1)+g_{13}P_{1,11}+g_{33}P_{2,11}+g_{33}P_{1,21}+g_{12}P_{1,12}+g_{23}P_{2,21}+g_{32}P_{2,12}+g_{32}P_{1,22}+g_{22}P_{2,22}\\
+2q_{13}u_{1,1}P_1+2q_{33}(u_{1,2}+u_{2,1})P_1+2q_{12}u_{1,1}P_2+2q_{23}u_{2,2}P_1+2q_{32}(u_{1,2}+u_{2,1})P_2+2q_{22}u_{2,2}P_2+f_{13}u_{1,11}\\
+f_{33}(u_{1,21}+u_{2,11})+f_{12}u_{1,12}+f_{23}u_{2,21}+f_{32}(u_{1,22}+u_{2,12})+f_{22}u_{2,22}-\varphi_{,2}=0(5)
\end{cases}$$

$unit:nN,nm,GPa,10^{-18}C,10^{-18}J$

PZT parameters:

$c_{11}=174.6;c_{12}=79.37;c_{13}=0.0;c_{22}=174.6;c_{23}=0.0;c_{33}=111.1;$

$q_{11}=0.089;11.41;q_{12}=-0.026; 0.461;q_{13}=0;q_{22}=0.089;11.41;q_{23}=0;q_{33}=0.032;7.5;$

$g_{11}=0.15;g_{12}=-0.15;g_{13}=0;g_{22}=0.15;g_{23}=0;g_{33}=0.15;$

$f_{11}=5;f_{12}=0;f_{13}=0;f_{22}=5;f_{23}=0;f_{33}=0;$

$\alpha_1=-0.148;\alpha_{11}=-0.031;\alpha_{12}=0.63;\alpha_{111}=0.25;\alpha_{112}=0.97;\epsilon_0=0.5841;L=1$

Additionally, some initial conditions and boundary conditions are known and can be used to help train the model.  

The boundary conditions are:

$$\varphi =0 $$

$$u_i = 0$$

$$P_{i,j}n_j=0$$

The initial conditions is:

$$P^2_1+P^2_2=0.2^2$$

