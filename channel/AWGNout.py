import numpy as np
from numpy import ones
from numpy import log
from numpy import exp
from numpy import real
from numpy import pi

class AWGNout: 
    # AwgnEstimOut:  AWGN scalar output estimation function
    #
    # Corresponds to an output channel of the form
    #   y = scale*z + N(0, wvar)

    # properties
    # y                  # measured output
    # wvar               # variance
    # scale = 1          # scale factor
    # maxSumVal = False  # True indicates to compute output for max-sum
    # autoTune = False   # Set to true for tuning of mean and/or variance
    # disableTune = False# Set to true to temporarily disable tuning
    # tuneMethod = 'Bethe'  # Tuning method, in {ML,Bethe,EM0,EM}
    # tuneDim = 'joint'  # Dimension to autoTune over, in {joint,col,row}
    # tuneDamp = 0.1     # Damping factor for autoTune in (0,1]
    # counter = 0        # Counter to delay tuning
    # wvar_min = 1e-20   # Minimum allowed value of wvar
    #end

    #methods

        # # Constructor
        # function self = AwgnEstimOut(y, wvar, maxSumVal, varargin)
    def __init__(self, y, wvar, maxSumVal=None):
        y                  # measured output
        wvar               # variance
        scale = 1          # scale factor


        self.y = y
        self.wvar = wvar
        if maxSumVal is not None:
            self.maxSumVal = maxSumVal
        else:
            self.maxSumVal = False  # True indicates to compute output for max-sum

        self.scale = 1          # scale factor
        self.maxSumVal = False  # True indicates to compute output for max-sum
        self.autoTune = False   # Set to true for tuning of mean and/or variance
        self.disableTune = False# Set to true to temporarily disable tuning
        self.tuneMethod = 'Bethe'  # Tuning method, in {ML,Bethe,EM0,EM}
        self.tuneDim = 'joint'  # Dimension to autoTune over, in {joint,col,row}
        self.tuneDamp = 0.1     # Damping factor for autoTune in (0,1]
        self.counter = 0        # Counter to delay tuning
        self.wvar_min = 1e-20   # Minimum allowed value of wvar



            # self = self@EstimOut;
        """
        if nargin ~= 0 # Allow nargin == 0 syntax
            self.y = y;
            self.wvar = wvar;
            if (nargin >= 3)
                if (~isempty(maxSumVal))
                    self.maxSumVal = maxSumVal;
                end
            end
            if (nargin >= 4)
                if isnumeric(varargin{1})
                    # make backwards compatible: 4th argument can specify the scale
                    self.scale = varargin{1};
                else
                    for i = 1:2:length(varargin)
                        self.(varargin{i}) = varargin{i+1};
                    end
                end
            end

            #Warn user about zero-valued noise variance
            #if any(self.wvar==0)
            #    warning(['Tiny non-zero variances will be used for'...
            #        ' computing log likelihoods. May cause problems'...
            #        ' with adaptive step size if used.']) ##ok<*WNTAG>
            #end
        end
        """


        # Set methods
    # def set_y(self, y)
    #     assert(all(imag(y(:)) == 0), ...
    #         'AwgnEstimOut: y must be real valued.  Did you mean to use CAwgnEstimOut instead?');
    #     # if we really want to handle real-valued noise and complex-valued y
    #     # (and thus complex z), then we need to modify this file!
    #     self.y = y;

    # def set_wvar(self, wvar)
    #     assert(all(wvar(:) >= 0), ...
    #         'AwgnEstimOut: wvar must be non-negative');
    #     self.wvar = wvar;

    # set_wvar_min(self, wvar_min)
    #     assert(all(wvar_min(:) > 0), ...
    #         'AwgnEstimOut: wvar_min must be positive');
    #     self.wvar_min = wvar_min;

    def maxSumVal(self, maxSumVal):
        # assert(isscalar(maxSumVal)&&(ismember(maxSumVal,[0,1])||islogical(maxSumVal)), ...
        #     'AwgnEstimOut: maxSumVal must be a logical scalar');
        self.maxSumVal = maxSumVal

    def set_scale(self, scale):
            # assert(isnumeric(scale)&&isscalar(scale)&&(scale>0), ...
            #     'AwgnEstimOut: scale must be a positive scalar');
        self.scale = scale

    def set_disableTune(self, flag):
            # assert(isscalar(flag)&&(ismember(flag,[0,1])||islogical(flag)), ...
            #     'AwgnEstimOut: disableTune must be a logical scalar');
        self.disableTune = flag


    # # Size  # FIXME
    # function [nz,ncol] = size(self)
    #     [nz,ncol] = size(self.y);
    # end


    def estim(self, pHat, pVar, nargout=2): #zHat, zVar, partition] = estim(self, pHat, pVar)
        """
        # AWGN estimation function
        # Provides the posterior mean and variance of _real_ variable z
        # from an observation real(y) = scale*z + w
        # where z = N(pHat,pVar) and w = N(0,wvar)
        """

        # Extract quantities
        y = self.y;
        scale = self.scale;
        pHat_real = real(pHat);
        scale2pVar = (scale**2)*pVar;

        # print("estim")
        # print(y.shape)
        # print(scale)
        # print(pVar.shape)
        # print(pVar)
        #
        # print(pHat_real.shape)
        # print(scale2pVar.shape)

        # Compute posterior mean and variance
        wvar = self.wvar;
        gain = pVar/(scale2pVar + wvar);
        zHat = np.multiply(scale*gain, y-scale*pHat_real) + pHat_real;
        zVar = wvar*gain;

        # print("shapes")
        # print(zHat.shape)
        # print(zVar.shape)

        # Tune noise variance
        if self.autoTune and  not self.disableTune:
            if (self.counter>0):  # don't tune yet
                self.counter = self.counter-1; # decrement counter
            else: # tune now
                M,T = pHat.shape
                damp = self.tuneDamp
                #Learn variance, averaging over columns and/or rows
                if self.tuneMethod == 'ML':
                    # argmax_wvar Z(y;wvar)=\int p(y|z;wvar) N(z;pHat,pVar) dz
                    if self.tuneDim == 'joint':
                        wvar1 = mean((scale*pHat_real-y)**2 - scale2pVar)
                        wvar0 = max(self.wvar_min, wvar1)*ones(self.wvar.shape)
                    elif self.tuneDim == 'col':
                        wvar1 = (1/M)*sum((scale*pHat_real-y)**2 - scale2pVar,1);
                        wvar0 = ones(M,1)*max(self.wvar_min, wvar1);
                    elif self.tuneDim == 'row':
                        wvar1 = (1/T)*sum((scale*pHat_real-y)**2 - scale2pVar,2);
                        wvar0 = max(self.wvar_min, wvar1)*ones(1,T);
                    else:
                        error('Invalid tuning dimension in AwgnEstimOut');

                    if damp==1:
                        self.wvar = wvar0
                    else: # apply damping
                        self.wvar = exp( (1-damp)*log(self.wvar) + damp*log(wvar0))

                elif self.tuneMethod == 'Bethe': # Method from Krzakala et al J.Stat.Mech. 2012
                    svar = 1/(scale2pVar + self.wvar);
                    shat = (y-scale*pHat_real)*svar;

                    if self.tuneDim == 'joint':
                        ratio = sum(shat**2)/sum(svar);
                        if damp is not 1:
                            ratio = ratio**damp
                        self.wvar = max(self.wvar_min, self.wvar*ratio);
                    elif self.tuneDim == 'col':
                        ratio = sum(shat**2,1)/sum(svar,1);
                        if damp is not 1:
                            ratio = ratio**damp
                        self.wvar = max(self.wvar_min, self.wvar*(ones(M,1)*ratio));
                    elif self.tuneDim ==  'row':
                        ratio = sum(shat**2,2)/sum(svar,2);
                        if damp is not 1:
                            ratio = ratio**damp
                        self.wvar = max(self.wvar_min, self.wvar*(ratio*ones(1,T)));
                    else:
                        error('Invalid tuning dimension in AwgnEstimOut');
                elif self.tuneMethod == 'EM0':
                    if self.tuneDim == 'joint':
                        wvar1 = mean((y-zHat)**2);
                        wvar0 = max(self.wvar_min, wvar1)*ones(size(self.wvar));
                    elif self.tuneDim == 'col':
                        wvar1 = (1/M)*sum((y-zHat)**2,1);
                        wvar0 = ones(M,1)*max(self.wvar_min, wvar1);
                    elif self.tuneDim == 'row':
                        wvar1 = (1/T)*sum((y-zHat)**2,2);
                        wvar0 = max(self.wvar_min, wvar1)*ones(1,T);
                    else:
                        error('Invalid tuning dimension in AwgnEstimOut');

                    if damp==1:
                        self.wvar = wvar0;
                    else: # apply damping
                        self.wvar = exp( (1-damp)*log(self.wvar) + damp*log(wvar0));
                elif self.tuneMethod == 'EM':
                    if self.tuneDim == 'joint':
                        wvar1 = mean((y-zHat)**2 + zVar);
                        wvar0 = max(self.wvar_min, wvar1)*ones(size(self.wvar));
                    elif self.tuneDim == 'col':
                        wvar1 = (1/M)*sum((y-zHat)**2 + zVar,1);
                        wvar0 = ones(M,1)*max(self.wvar_min, wvar1);
                    elif self.tuneDim == 'row':
                        wvar1 = (1/T)*sum((y-zHat)**2 + zVar,2);
                        wvar0 = max(self.wvar_min, wvar1)*ones(1,T);
                    else:
                        error('Invalid tuning dimension in AwgnEstimOut');

                    if damp==1:
                        self.wvar = wvar0;
                    else: # apply damping
                        self.wvar = exp( (1-damp)*log(self.wvar) + damp*log(wvar0));
                else:
                    error('Invalid tuning method in AwgnEstimOut')

        # end of (if self.autoTune && ~self.disableTune:)

        # Compute partition function
        if nargout==3:
            partition = (1/sqrt(2*pi*(scale2pVar+wvar))) * exp(-(0.5*(pHat_real-y)**2) / (scale2pVar+wvar) )
            return zHat, zVar, partition
        else:
            return zHat, zVar

    # end of function





    def logLike(self,pHat,pVar):

        # Compute output cost:
        # For sum-product compute
        #   E_Z( log p_{Y|Z}(y|z) ) with Z ~ N(pHat, pVar)
        # For max-sum GAMP, compute
        #   log p_{Y|Z}(y|z) @ z = pHat

        # Ensure variance is small positive number
        wvar_pos = max(self.wvar_min, self.wvar)

        # Get scale
        scale = self.scale

        # Compute log-likelihood
        if ~(self.maxSumVal):
            predErr = ((self.y-scale*real(pHat))**2 + (scale**2)*pVar)/wvar_pos
        else:
            predErr = ((self.y-scale*real(pHat))**2)/wvar_pos
        end
        ll = -0.5*(predErr); #return the values without summing

        return ll




    def logScale(self,Axhat,pVar,pHat,alpha=1):  ##ok<INUSL>

        # Compute output cost:
        #   (Axhat-pHatfix)^2/(2*pVar*alpha) + log int_z p_{Y|Z}(y|z) N(z;pHatfix, pVar)
        #   with pHatfix such that Axhat=alpha*estim(pHatfix,pVar) + (1-alpha)*pHatfix.
        # For max-sum GAMP, compute
        #   log p_{Y|Z}(y|z) @ z = Axhat

        # #Set alpha if not provided to unity
        # if nargin < 5
        #     alpha = 1;
        # end

        # Ensure variance is small positive number
        wvar1 = max(self.wvar_min, self.wvar)

        #Get the scale
        s = self.scale

        # Compute output cost
        if ~(self.maxSumVal):

            #Use closed form
            closed_form = True;

            # Compute output cost
            if alpha is not 1:   #if any(alpha is not 1):
                alphaFlag = True;
            else:
                alphaFlag = False;

            if closed_form:
                #ptil = (pVar./wvar1+1).*Axhat - (pVar./wvar1).*self.y;
                #Old closed form update without scale
                #                     ll = -0.5*( log(pVar+wvar1) + (self.y-real(Axhat)).^2./wvar1 ...
                #                         + log(2*pi) );

                # Compute in closed form
                if ~alphaFlag:
                    # print(pVar.shape)
                    # print(self.y.shape)

                    ll = -0.5*( log(s**2 * pVar + wvar1) + np.square(self.y - s*real(Axhat))/wvar1 + log(2*pi));
                    # print("shape of calculated ll")
                    # print(ll.shape)
                    #print(wvar1.shape) # wvar1 is float
                    # print(Axhat.shape)
                    # print(self.y.shape)
                else:
                    ll = -0.5*( log(s**2*pVar + wvar1) + np.divide(np.square(self.y - s*real(Axhat)),wvar1 + pVar*(1-alpha)) + log(2*pi));


            else:
                # Find the fixed-point of pHat
                opt.pHat0 = Axhat; # works better than pHat
                opt.alg = 1; # approximate newton's method
                opt.maxIter = 3;
                opt.tol = 1e-4;
                opt.stepsize = 1;
                opt.regularization = self.wvar**2;  # works well up to SNR=160dB
                opt.debug = false;
                opt.alpha = alpha;
                [pHatfix,zHat] = estimInvert(self,Axhat,pVar,opt);

                # Compute log int_z p_{Y|Z}(y|z) N(z;pHatfix, pVar)
                ls = -0.5*log(2*pi*(self.wvar + s**2*pVar)) - np.divide(np.square(self.y - s*real(pHatfix)) , 2*(self.wvar + s**2*pVar));

                # Combine to form output cost
                ll = ls + 0.5*np.divide(np.square(real(zHat - pHatfix)),pVar)*alpha

        else:
            # Output cost is simply the log likelihood
            ll = -0.5*np.divide(np.square(self.y-s*real(Axhat)), wvar1);


        return ll



    def numColumns(self):
        #Return number of columns of Y
        #S = size(self.y,2);
        shape = self.y.shape
        if len(shape)>1:
            return shape[1]
        else:
            return 1

    # Generate random samples from p(y|z)
    def genRand(self, z):
        y = sqrt(self.wvar)*randn(size(z)) + self.scale*z;
        return y
