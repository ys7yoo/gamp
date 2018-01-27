import numpy as np
from numpy import real
from numpy import log

class AWGNin:
    """AWGN scalar input estimation def"""


    ## attributes
    #var0_min = eps;     # Minimum allowed value of var0
    var0_min = 1e-12
    mean0 = 0;          # Prior mean
    var0 = 1;           # Prior variance
    maxSumVal = False;  # True indicates to compute output for max-sum
    autoTune = False;   # Set to true for taut tuning of params
    disableTune = False;# Set to true to temporarily disable tuning
    mean0Tune = True;   # Enable Tuning of mean0
    var0Tune = True;    # Enable Tuning of var0
    tuneDim = 'joint';  # Determine dimension to autoTune over, in {joint,col,row}
    counter = 0;        # Counter to delay tuning

    # attributes (Hidden)
    mixWeight = 1;              # Weights for autoTuning

    ## methods
    # Constructor #FIXME
    def __init__(self, mean0=None, var0=None, maxSumVal=None):
        self.mean0 = mean0
        self.var0 = var0
        self.maxSumVal = maxSumVal

    """
        #obj = obj@EstimIn;
        if nargin ~= 0: # Allow nargin == 0 syntax
            self.mean0 = mean0;
            self.var0 = var0;
            if (nargin >= 3):
                if (~isempty(maxSumVal)):
                    self.maxSumVal = maxSumVal;
            for i = 1:2:length(varargin)
                self.(varargin{i}) = varargin{i+1};
            # warn user about inputs
            #if any(~isreal(mean0(:))),
            #    error('First argument of AwgnEstimIn must be real-valued');
            #end;
            #if any((var0(:)<0))||any(~isreal(var0(:))),
            #    error('Second argument of AwgnEstimIn must be non-negative');
            #end;
    """


    #Set Methods
    def set_var0_min(self, var0_min):
#        assert(all(var0_min(:) > 0), ...
#            'AwgnEstimIn: var0_min must be positive');
        self.var0_min = var0_min

    def set_mean0(self, mean0):
#        assert(all(isreal(mean0(:))), ...
#            'AwgnEstimIn: mean0 must be real-valued');
        self.mean0 = mean0

    def set_var0(self, var0):
#        assert(all(var0(:) > 0), ...
#            'AwgnEstimIn: var0 must be positive');
        self.var0 = max(self.var0_min,var0) # avoid too-small variances!

    def set_mixWeight(self, mixWeight):
#        assert(all(mixWeight(:) >= 0), ...
#            'AwgnEstimIn: mixWeights must be non-negative');
        self.mixWeight = mixWeight

    def set_maxSumVal(self, maxsumval):
#        assert(isscalar(maxsumval)&&(ismember(maxsumval,[0,1])||islogical(maxsumval)), ...
#            'AwgnEstimIn: maxSumVal must be a logical scalar');
        self.maxSumVal = maxsumval

    def set_disableTune(self, flag):
#        assert(isscalar(flag)&&(ismember(flag,[0,1])||islogical(flag)), ...
#            'AwgnEstimIn: disableTune must be a logical scalar');
        self.disableTune = flag

    # Prior mean and variance
    def estimInit(self):
        mean0 = self.mean0
        var0  = self.var0
        valInit = 0.0
        return mean0, var0, valInit

    # Size
    def size(self):
        return size(self.mean0)

    # AWGN estimation function
    # Provides the mean and variance of a variable x = N(qHat0,qVar0)
    # from an observation real(rHat) = x + w, w = N(0,rVar)
    def estim(self, rHat, rVar=None, nargout=2):
        # Get prior
        qHat0 = self.mean0
        qVar0 = self.var0

        # Compute posterior mean and variance
        gain = qVar0/(qVar0+rVar)
        xhat = np.multiply(gain, real(rHat)-qHat0)+qHat0
        xvar = np.multiply(gain, rVar)

        if self.autoTune and not self.disableTune:

          if (self.counter>0): # don't tune yet
            self.counter = self.counter-1 # decrement counter
          else: #  tune now
            """
            print size(rHat)
            [N, T] = size(rHat)
            # Learn mean if enabled
            if self.mean0Tune:
                # Average over all elements, per column, or per row
                if self.tuneDim == 'joint':
                    print self.mixWeight
                    print xhat
                    print "self.mean0 = sum(self.mixWeight(:)*xhat(:))/N/T"
                elif self.tuneDim == 'col':
                    print "self.mean0 = repmat(sum(self.mixWeight*xhat)/N, [N 1])"
                elif self.tuneDim =='row':
                    print "#self.mean0 = repmat(sum(self.mixWeight*xhat,2)/T, [1 T])"
                else:
                    error('Invalid tuning dimension in AwgnEstimIn')

            # Learn variance if enabled
            if self.var0Tune:
                #Average over all elements, per column, or per row
                if self.tuneDim == 'joint':
                    self.var0 = sum(self.mixWeight(:)
                            *(xhat(:) - self.mean0(:))**2 + xvar(:))/(N*T)
                elif self.tuneDim == 'col':
                    self.var0 = repmat(sum(self.mixWeight
                            *(xhat - self.mean0)**2 + xvar, 1)/N, [N 1])
                elif self.tuneDim == 'row'
                    self.var0 = repmat(sum(self.mixWeight
                            *(xhat - self.mean0)**2 + xvar, 2)/T, [1 T])
                else:
                    error('Invalid tuning dimension in AwgnEstimIn')
                #qVar0 = max(self.var0_min,self.var0);
            """
        if rVar is not None:
            if ~(self.maxSumVal):
                #Compute the negative KL divergence
                #   klDivNeg = \sum_i \int p(x|r)*\log( p(x) / p(x|r) )dx
                xvar_over_qVar0 = rVar/(qVar0+rVar)
                val = 0.5*(log(xvar_over_qVar0) + (1.0-xvar_over_qVar0)
                    - np.square(xhat-qHat0)/qVar0 )
            else:
                # Evaluate the (log) prior
                val = -0.5*np.square(xhat-qHat0)/qVar0
        return xhat, xvar, val

    # Generate random samples
    def genRand(self, outSize):
        if isscalar(outSize):
            x = sqrt(self.var0)*randn(outSize,1) + self.mean0
        else:
            x = sqrt(self.var0)*randn(outSize) + self.mean0
        return x

    # Computes the likelihood p(rHat) for real(rHat) = x + v, v = N(0,rVar)
    def plikey(self,rHat,rVar):
        py = exp(-1/(2*(self.var0+rVar))*(real(rHat)-self.mean0)**2)
        py = py/ sqrt(2*pi*(self.var0+rVar))
        return py

    # Computes the log-likelihood, log p(rHat), for real(rHat) = x + v, where
    # x = N(self.mean0, self.var0) and v = N(0,rVar)
    def loglikey(self, rHat, rVar):
        logpy = (-0.5)*( log(2*pi) + log(self.var0 + rVar) +
            ((real(rHat) - self.mean0)**2) / (self.var0 + rVar) )
        return logpy
