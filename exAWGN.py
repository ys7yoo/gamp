import numpy as np
import time

# export PYTHONPATH=.
from channel.AWGNin import AWGNin
from channel.AWGNout import AWGNout
from gamp.Option import GampOption
from gamp.estimate  import estimate

# Simple example of estimating a Gaussian vector.

# In this problem, x is a Gaussian random vector that we want to
# estimate from measurements of the form

#   y = A*x + w,

# where A is a random matrix and w is Gaussian noise.  This is a classical
# LMMSE estimation problem and can be easily performed in MATLAB
# without the GAMP method.  But, you can look at this example, just to
# understand the syntax of the gampEst function.

##  Handle random seed [FIXME]


# Parameters
nx=100
nz=200
snr=100

# Create a random Gaussian vector
xmean0=0.0
xvar0=1.0

# Compute the noise level based on the specified SNR. Since the components
# of A have power 1/nx, the E|y(i)|^2 = E|x(j)|^2 = xmean^2+xvar.
wvar = 10**(-0.1*snr)*(xmean0**2+xvar0)     # 1e-10
print(wvar)


DEBUG = True
if DEBUG:
    # load data generated from Matlab
    # ln -s ../gamp-matlab/code/examples/basic/A.txt A.txt
    # ln -s ../gamp-matlab/code/examples/basic/x.txt x.txt
    # ln -s ../gamp-matlab/code/examples/basic/w.txt w.txt
    # ln -s ../gamp-matlab/code/examples/basic/y.txt y.txt

    A=np.loadtxt('A.txt')
    x=np.loadtxt('x.txt')
    w=np.loadtxt('w.txt')
    y=np.loadtxt('y.txt')

    A=np.matrix(A)
    x=np.matrix(x).transpose()
    w=np.matrix(w).transpose()
    y=np.matrix(y).transpose()
    # print(A)
    # print(x)
    # print(w)
    # print(y)
else: # generate  data

    # Create a random measurement matrix
    A=(1 / sqrt(nx))*np.random.randn(nz,nx)
    #print(A)

    #x=dot(xmean0, ones(nx,1)) + dot(sqrt(xvar0),randn(nx,1))
    x=xmean0*np.ones((nx,1)) + np.sqrt(xvar0)*np.random.randn(nx,1)
    #print(x)

    # Generate the noise
    w=sqrt(wvar)*np.random.randn(nz,1)
    #print(w)
    y=dot(A,x) + w


print(A.shape)
print(x.shape)
print(w.shape)
print(y.shape)

# Decide on MAP or MMSE GAMP
map=0

# Create an input estimation class corresponding to a Gaussian vector
ChannelIn=AWGNin(xmean0,xvar0,map)

# Create an output estimation class corresponding to the Gaussian noise.
# Note that the observation vector is passed to the class constructor, not
# the gampEst function.
ChannelOut=AWGNout(y,wvar,map)

# Set the default options
opt=GampOption()
opt.nit = 10000
opt.tol = max(min(0.001, 10**(- snr / 10)), 1e-15)
opt.uniformVariance = 0
opt.pvarMin = 0
opt.xvarMin = 0
opt.adaptStep = True
opt.adaptStepBethe = True
opt.legacyOut = False
# Demonstrate automatic selection of xvar0
if 0:
    opt.xvar0auto = True
    opt.xHat0 = copy(x + dot(dot(0.01,randn(nx,1)),norm(x)) / sqrt(nx))

## Run the GAMP algorithm
tic = time.time()
#estFin,optFin,estHist=gampEst(ChannelIn,ChannelOut,A,opt) #,nargout=3)
#xHat=estFin.xHat

xHat,opt=estimate(ChannelIn,ChannelOut,A,opt)  # simpler output (for now)
toc =  time.time()
timeGAMP=toc-tic

print(xHat)

#xHat = np.matrix(xHat)
#np.save
np.savetxt("xHatPy.txt",xHat)
# with open('XhatPy.txt','w') as f:
#     for line in xHat:
#         np.savetxt(f, line) #, fmt='%.2f')




# Now perform the exact LMMSE solution
tic = time.time()
AA=np.dot(A.T,A) + wvar/xvar0*np.eye(nx)
BB=np.dot(A.T, y - np.dot(A,np.ones((nx,1))*xmean0))
xHatLMMSE=xmean0 + np.linalg.solve(AA, BB)
# xHatLMMSE = xmean0 + (A'*A + wvar/xvar0*eye(nx))\(A'*(y-A*ones(nx,1)*xmean0));
toc =  time.time()
timeLMMSE=toc-tic

np.savetxt("xHatLMMSEPy.txt",xHat)


print("errors: ")
print((xHat-x).T*(xHat-x)/len(x))
print((xHatLMMSE-x).T*(xHatLMMSE-x)/len(x))



exit()




import matplotlib.pyplot as plt

fig1 = plt.figure()
plt.plot(x,xHat, 'g.', x,xHatLMMSE, 'r.')



## Plot the results
figure(1)
clf
xsort,I=sort(x,nargout=2)
handy=plot(xsort,xsort,'-',xsort,xHat[I],'g.',xsort,xHatLMMSE[I],'r.')
#set(handy(2),'MarkerSize',8);
#set(handy(3),'MarkerSize',8);
set(gca,'FontSize',16)
grid('on')
legend('True','GAMP estimate','LMMSE estimate')
xlabel('True value of x')
ylabel('Estimate of x')
figure(2)
clf
subplot(311)
plot(dot(10,log10(sum(abs(estHist.xHat - dot(x,ones(1,size(estHist.xHat,2)))) ** 2,1) / norm(x) ** 2)))
ylabel('NMSE [dB]')
grid('on')
subplot(312)
plot(estHist.step)
ylabel('step')
grid('on')
subplot(313)
plot(estHist.val)
ylabel('val')
xlabel('iteration')
grid('on')
# Display the MSE
mseGAMP=dot(20,log10(norm(x - xHat) / norm(x)))
mseLMMSE=dot(20,log10(norm(x - xHatLMMSE) / norm(x)))
fprintf(1,'GAMP: MSE = %5.5f dB\\n',mseGAMP)
fprintf(1,'LMMSE:   MSE = %5.5f dB\\n',mseLMMSE)
