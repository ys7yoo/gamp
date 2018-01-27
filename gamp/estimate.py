# Autogenerated with SMOP
#from smop.core import *
# gampEst.m

from numpy import inf
from numpy import nan
from numpy import isnan
#from numpy import max
#from numpy import min

from numpy import ones
from numpy import zeros
from numpy import isscalar

eps = 1e-12


from numpy.linalg import norm


#from inspect import ismethod
#
def ismethod(obj, method_name):
    # code from https://stackoverflow.com/questions/5963729/check-if-a-function-is-a-method-of-some-object
    return hasattr(obj, method_name) and callable(getattr(obj, method_name))

import numpy as np

def multSq(A,x):
    Asq = np.square(np.abs(A))    # component-wise square
    return np.dot(Asq,x)

def printMat(A, name):
    shape=A.shape
    print(name+str(shape)+"=")
    if shape[1]==1:
        print(A.transpose())
    else:
        print(A)


def estimate(scaEstIn=None,scaEstOut=None,A=None,opt=None,saveHist=None):
    # varargin = gampEst.varargin
    # nargin = gampEst.nargin

    # gampEst:  Generalized Approximate Message passeding -- Estimation algorithm

    # DESCRIPTION:
# ------------
# The G-AMP estimation algorithm is intended for the estimation of a
# random vector x observed through an observation y from the Markov chain

    #   x -> z = A*x -> y,

    # where the prior p(x) and likelihood function p(y|z) are both separable.

    # SYNTAX:
# -------
# [out1,out2,out3,out4,out5,out6,out7,out8,out9] = ...
#                               gampEst(scaEstIn, scaEstOut, A, opt)

    # INPUTS:
# -------
# scaEstIn:  An input estimator derived from the EstimIn class
#    based on the input distribution p_X(x_j).
# scaEstOut:  An output estimator derived from the EstimOut class
#    based on the output distribution p_{Y|Z}(y_i|z_i).
# A:  Either a matrix or a linear operator defined by the LinTrans class.
# opt:  A set of options of the class GampOpt.

    # OUTPUTS:
# --------
#  LEGACY FORMAT:
#  out1 = xHatFinal
#  out2 = xVarFinal [optional]
#  out3 = rHatFinal [optional]
#  out4 = rVarFinal [optional]
#  out5 = sHatFinal [optional]
#  out6 = sVarFinal [optional]
#  out7 = zHatFinal [optional]
#  out8 = zVarFinal [optional]
#  out9 = estHist [optional]

    #  NEW FORMAT:
#  out1 = estFin
#  out2 = optFin
#  out3 = estHist [optional]

    #   ... where ...

    #  xHatFinal: final estimate of the vector x (output of x-estimator).
#  xVarFinal: final quadratic term for vector x (output of x-estimator).
#  pHatFinal: final estimate of the vector p (input to z-estimator).
#  pVarFinal: final quadratic term for vector p (input to z-estimator).
#  zHatFinal: final estimate of the vector z=Ax (output of z-estimator).
#  zVarFinal: final quadratic term for vector z=Ax (output of z-estimator).
#  sHatFinal: final estimate of the vector s (lagrange penalty on z-Ax).
#  sVarFinal: final quadratic term for vector s (lagrange penalty on z-Ax).
#  rHatFinal: final estimate of the vector r (input to x-estimator).
#  rVarFinal: final quadratic term for vector r (input to x-estimator).

    # estFin:  Final G-AMP estimation quantities
#   .xHat: same as xHatFinal above
#   .xVar: same as xVarFinal above
#   .AxHat: same as A*xHatFinal above
#   .pHat: same as pHatFinal above
#   .pVar: same as pVarFinal above
#   .zHat: same as zHatFinal above
#   .zVar: same as zVarFinal above
#   .sHat: same as sHatFinal above
#   .sVar: same as sVarFinal above
#   .rHat: same as rHatFinal above
#   .rVar: same as rVarFinal above
#   .xHatPrev: previous iteration of xHatFinal
#   .xHatNext: next iteration of xHat (used for warm start)
#   .xVarNext: next iteration of xVar (used for warm start)
#   .xHatDamp: damping state on xHat (used for warm start)
#   .pVarOpt = damping state on pVar (used for warm start)
#   .rVarOpt = damping state on rVar (used for warm start)
#   .A2xVarOpt = damping state on A2xVar (used for warm start)
#   .sHatNext: next iteration of sHat (used for warm start)
#   .sVarNext: next iteration of sVar (used for warm start)
#   .val: final value of utility (i.e., negative cost)
#   .valIn: final value of input utility (i.e., negative cost)
#   .valOpt: final record of input utilities (i.e., negative cost)
#   .scaleFac: final value of scaling used when varNorm=true
#   .step: final value of the stepsize (i.e., damping term)
#   .stepMax: final value of the maximum stepsize

    # optFin:  Final settings of GampOpt options object (see GampOpt.m)

    # estHist:  History of G-AMP across iterations
#   .xHat: history of xHat
#   .xVar: history of xVar
#   .AxHat: history of A*xHat
#   .pHat: history of pHat
#   .pVar: history of pVar
#   .zHat: history of zHat
#   .zVar: history of zVar
#   .sHat: history of sHat
#   .sVar: history of sVar
#   .rHat: history of rHat
#   .rVar: history of rVar
#   .passed: history of passed/fail
#   .val: history of the utility (i.e., negative cost)
#   .scaleFac: history of the scalefactor used when varNorm=true
#   .step: history of the stepsize (i.e., damping term)
#   .stepMax: history of the maximum allowed stepsize
#   .it = lists the iterations reported in the history

    # Note that, in sum-product mode, the marginal posterior pdfs are
#    p(x(j)|y) ~= Cx*p(x(j))*exp( -(x(j)-rHat(j))^2/(2*rVar(j) )
#    p(z(i)|y) ~= Cz*p(y(i)|z(i))*exp( -(z(i)-pHat(i))^2/(2*pVar(i) )
# where Cx and Cz are normalization constants.

    # Get options
    if opt==None:
        opt=GampOpt()

    nit=opt.nit

    step=opt.step

    stepMin=opt.stepMin

    stepMax=opt.stepMax

    stepIncr=opt.stepIncr

    stepDecr=opt.stepDecr

    adaptStep=opt.adaptStep

    adaptStepBethe=opt.adaptStepBethe

    stepWindow=opt.stepWindow

    bbStep=opt.bbStep

    verbose=opt.verbose

    tol=opt.tol

    maxBadSteps=opt.maxBadSteps

    maxStepDecr=opt.maxStepDecr

    stepTol=opt.stepTol

    pVarStep=opt.pVarStep

    rVarStep=opt.rVarStep

    varNorm=opt.varNorm

    scaleFac=opt.scaleFac

    pVarMin=opt.pVarMin

    rVarMin=opt.xVarMin

    zVarTopVarMax=opt.zVarTopVarMax

    histIntvl=opt.histIntvl

    # Handle output format
    legacyOut=opt.legacyOut

    # if (legacyOut):
    #     saveHist=(nargout >= 9)
    #     if (nargout > 9):
    #         error('too many output arguments')
    # else:
    #     saveHist=(nargout >= 3)
    #     if (nargout > 3):
    #         error('too many output arguments')


    # Determine whether the utility must be computed each iteration
    if adaptStep:
        compVal=True
    else:
        compVal=False

    # Check for the presence of a custom stopping criterion in the options
# structure, and set flags as needed
    if opt.stopFcn: #not empty   # not isempty(opt.stopFcn):
        customStop=1
        stopFcn=opt.stopFcn
    else:
        if opt.stopFcn2: #not empty   # not isempty(opt.stopFcn2):
            customStop=2
            stopFcn2=opt.stopFcn2
        else:
            customStop=False

    # If A is an explicit matrix, replace by an operator
    # if isa(A,'double'):           # FIXME
    #     A=MatrixLinTrans(A)

    # Get dimensions
    #m,n=A.size(nargout=2)
    m,n=A.shape
    s=scaEstOut.numColumns()
    # Get default initialization values
    #xHat,xVar,valIn=scaEstIn.estimInit(nargout=3)
    xHat,xVar,valIn=scaEstIn.estimInit()
    # print('xHat=')
    # print(xHat)
    # print('xVar=')
    # print(xVar)
    # print('valIn=')
    # print(valIn)
    if not isinstance(valIn, float):
        #valIn=sum(ravel(valIn))  # valIn = sum( valIn(:) );
        valIn=sum(valIn)
    # Replace default initialization with user-provided values
    if opt.xHat0:        #logical_not(isempty(opt.xHat0)):
        if sum(xHat != opt.xHat0):
            valIn = - inf
        xHat=opt.xHat0

    if opt.xVar0:        #logical_not(isempty(opt.xVar0)):
        if sum(xVar != opt.xVar0):
            valIn = - inf
        xVar=opt.xVar0

    #valIn = -Inf; # only for backwards compatibility.  Remove in next major revision?
    if opt.valIn0:      #logical_not(isempty(opt.valIn0)):
        valIn=opt.valIn0

    #valOpt = []     # empty initialization will cause the first iteration to be a "passed"
    valOpt = [-inf]
    if opt.valOpt0:         #logical_not(isempty(opt.valOpt0)):
        valOpt=opt.valOpt0

    val= nan
    # For a scalar output, the same distribution is applied to all components
    if isscalar(xHat):  #(size(xHat,1) == 1):
        xHat=xHat*ones((n,1))
    if isscalar(xVar):  #(size(xVar,1) == 1):
        xVar=xVar*ones((n,1))
    # print('xHat=')
    # print(xHat)
    # print('xVar=')
    # print(xVar)

    # Make sure that number of input columns match number of output columns
    foo, numCol = xHat.shape
    #if (size(xHat,2) == 1):
    if numCol == 1:
        xHat=xHat*ones((1,s))

    foo, numCol = xVar.shape
    #if (size(xVar,2) == 1):
    if numCol == 1:
        xVar=xVar*ones((1,s))
    # print('xHat=')
    # print(xHat)
    # print('xVar=')
    # print(xVar)

    # Continue with initialization
    sHat=zeros((m,s))
    sVar=nan*ones((m,s))
    xHatDamp=nan*ones((m,s))
    pVarOpt=nan*ones((m,s))
    rVarOpt=nan*ones((m,s))
    A2xVarOpt=nan*ones((m,s))

    # Replace default initialization with user-provided values
    if opt.sHat0:       # logical_not(isempty(opt.sHat0)):
        sHat=dot(opt.sHat0,scaleFac)
    if opt.sVar0:       # logical_not(isempty(opt.sVar0)):
        sVar=dot(opt.sVar0,scaleFac)
    if opt.xHatPrev0:       # logical_not(isempty(opt.xHatPrev0)):
        xHatDamp=opt.xHatPrev0
    if opt.pVarOpt0:        # logical_not(isempty(opt.pVarOpt0)):
        pVarOpt=opt.pVarOpt0
    if opt.rVarOpt0:         #logical_not(isempty(opt.rVarOpt0)):
        rVarOpt=opt.rVarOpt0
    if opt.A2xVarOpt0:      #logical_not(isempty(opt.A2xVarOpt0)):
        A2xVarOpt=opt.A2xVarOpt0
    # Replace the stepMax adaptation quantities with user-provided values
    failCount=0
    if opt.failCount0:      # logical_not(isempty(opt.failCount0)):
        failCount=opt.failCount0

    # If the mean-removal option is set, create an augmented system
    # with the mean removed.  (See LinTransDemeanRC.m for more details.)
    if (opt.removeMean):
        A=LinTransDemeanRC(A,opt.removeMeanExplicit)
        #m,n=A.size(nargout=2)
        m,n=A.shape
        maxSumVal=False
        isCmplx=False
        scaEstOut=A.expandOut(scaEstOut,maxSumVal,isCmplx)
        scaEstIn=A.expandIn(scaEstIn)
        xHat=A.expandxHat(xHat)
        xVar=A.expandxVar(xVar)
        sHat=A.expandsHat(sHat)
        sVar=A.expandsVar(sVar)
        xHatDamp=A.expandxHat(xHatDamp)
        pVarOpt=A.expandsVar(pVarOpt)
        rVarOpt=A.expandxVar(rVarOpt)
        A2xVarOpt=A.expandsVar(A2xVarOpt)

    # If uniform-variance mode is requested by the user, implement it by
    # redefining the A.multSq and A.multSqTr operations
    if (opt.uniformVariance):
        if not opt.removeMean:   #logical_not((opt.removeMean)):
            A=UnifVarLinTrans(A)
        else:
            A=UnifVarLinTrans(A,arange(1,m - 2),arange(1,n - 2))

    # If desired, automatically set xVar
    if (opt.xVar0auto):
        print("automatically set xVar")
        # temporarily disable autoTuning
        if any(strcmp('disableTune',properties(scaEstOut))):
            disOut=scaEstOut.disableTune
            scaEstOut.disableTune = copy(true)
        if any(strcmp('disableTune',properties(scaEstIn))):
            disIn=scaEstIn.disableTune
            scaEstIn.disableTune = copy(true)
        # setup estimInvert options for both z & x variables
        xVarTol=0.0001
        zopt.maxIter = 100
        zopt.stepsize = 0.25
        zopt.regularization = 1e-20
        zopt.tol = 0.0001
        zopt.debug = False
        xopt=copy(zopt)
        # iterate to find fixed-point xVar
        xHat0=copy(xHat)
        AxHat0=np.dot(A,xHat0)
        for t in arange(100).reshape(-1):
            pVar=max(pVarMin,multSq(A,xVar))
            pHat,zHat,zVar,zstep=estimInvert(scaEstOut,AxHat0,pVar,zopt,nargout=4)
            zopt.stepsize = copy(zstep)
            zopt.pHat0 = copy(pHat)
            #NRz = norm(zHat(:)-AxHat0(:))/norm(AxHat0(:))
            sVar=(1 - zVar / pVar) / pVar
            sVar[abs(sVar) < eps]=eps
            rVar=max(rVarMin,1.0 / (A.multSqTr(sVar)))
            xVarOld=copy(xVar)
            rHat,xHat,xVar,xstep=estimInvert(scaEstIn,xHat0,rVar,xopt,nargout=4)
            xopt.stepsize = copy(xstep)
            xopt.pHat0 = copy(rHat)
            #NRx = norm(xHat(:)-xHat0(:))/norm(xHat0(:))
            if norm(ravel(xVar) - ravel(xVarOld)) < dot(norm(ravel(xVar)),xVarTol):
                break
        xHat=copy(xHat0)
        if any(strcmp('disableTune',properties(scaEstOut))):
            scaEstOut.disableTune = copy(disOut)
        if any(strcmp('disableTune',properties(scaEstIn))):
            scaEstIn.disableTune = copy(disIn)

    # Declare variables
    zHat=nan*ones((m,s))
    zVar=nan*ones((m,s))
    pHat=nan*ones((m,s))
    rHat=nan*ones((m,s))
    rVar=nan*ones((m,s))
    xHatFinal=nan*ones((m,s))
    if (saveHist):
        nitSave=floor(nit / histIntvl)
        estHist.xHat = nan(n*s,nitSave)
        estHist.xVar = nan(n*s,nitSave)
        estHist.AxHat = nan(m*s,nitSave)
        estHist.pHat = nan(m*s,nitSave)
        estHist.pVar = nan(m*s,nitSave)
        estHist.sHat = nan(m*s,nitSave)
        estHist.sVar = nan(m*s,nitSave)
        estHist.zHat = nan(m*s,nitSave)
        estHist.zVar = nan(m*s,nitSave)
        estHist.rHat = nan(n*s,nitSave)
        estHist.rVar = nan(n*s,nitSave)
        estHist.step = nan(nitSave,1)
        estHist.val = nan(nitSave,1)
        estHist.stepMax = nan(nitSave,1)
        estHist.passed = nan(nitSave,1)
        estHist.scaleFac = nan(nitSave,1)

    # Check for the presence of two methods within the LinTrans and EstimIn
    # objects and set flags accordingly
    MtxUncertaintyFlag=ismethod(A,'includeMatrixUncertainty')
    MsgUpdateFlag=ismethod(scaEstIn,'msgUpdate')
    # If using BB stepsize adaptation, compute column norms for use in scaling
    if bbStep:
        columnNorms=A.multSqTr(ones(m,1)) ** 0.5
        columnNorms=repmat(columnNorms,1,s)

    # Control variables to terminate the iterations
    stop=False
    it=0


    ###########################################################################
    ## Main iteration loop
    ###########################################################################
    while not stop:         #logical_not(stop):

        # Iteration count
        it=it + 1
        if it >= nit:
            stop=True
        print("itr="+str(it))

        # Check whether to save this iteration in history
        if saveHist and rem(it,histIntvl) == 0:
            itSaveHist=it / histIntvl
        else:
            itSaveHist=[]

        ########################################################################
        # 2. Output linear stage: find pHat and pVar (update prior on Z ~ N(pHat,pVar))
        # print("2. Output linear stage")
        # with no A uncertainty
        #A2xVar=multSq(A,xVar)
        A2xVar = multSq(A,xVar)
#        print(A.shape)
        #print(xVar.shape)
        #print(A2xVar.shape)
        # printMat(A2xVar,"A2xVar")

        if MtxUncertaintyFlag:
            print("Include Matrix Uncertainty")
            pVar=A.includeMatrixUncertainty(A2xVar,xHat,xVar)
        else:
            pVar=A2xVar

        # Continued output linear stage
        AxHat=np.dot(A,xHat)

        # Step in pVar
        if pVarStep:
            if (it == 1):
                if any(isnan(pVarOpt)):
                    pVarOpt=pVar
                if any(isnan(A2xVarOpt)):
                    A2xVarOpt=A2xVar
            pVar = np.dot((1.0 - step),pVarOpt) + np.dot(step,pVar)
            A2xVar = np.dot((1.0 - step),A2xVarOpt) + np.dot(step,A2xVar)

        ## Continued output linear stage
        pHat = AxHat - 1.0 / scaleFac * np.multiply(A2xVar,sHat) ## Eq. 9b % Note: uses A2xVar rather than pVar WHY?
        pVarRobust = np.maximum(pVar,pVarMin)   # At very high SNR, use very small pVarMin!
        # # DEBUGGING ---------
        # printMat(pHat, 'pHat')
        # printMat(pVar, 'pVar')
        # printMat(pVarRobust, 'pVarRobust')
        # # --------------------


        ## Compute expected log-likelihood of the output and add to negative
        # KL-divergence of the input, giving the current utility function
        if (compVal):
            #print(adaptStepBethe)  True
            if not adaptStepBethe: #logical_not(adaptStepBethe):
                valOut=sum(sum(scaEstOut.logLike(AxHat,pVar)))
            else:
                # print("calc ll")
                ll = scaEstOut.logScale(AxHat,pVar,pHat)
                # print(ll.shape)
                valOut=sum(sum(ll))
            val=valOut + valIn

            print('val='+str(val))
            # print("valOut=")
            # print(valOut)
            # print("valIn=")
            # print(valIn)
        # An iteration "passedes" if any of below is true:
        # 1. Adaptive stepsizing is turned off
        # 2. Current stepsize is so small it can't be reduced
        # 3. The current utility at least as large as the worst in the stepWindow
        # Also, we force a passed on the first iteration else many quantities undefined

        stopInd=len(valOpt)
        startInd=max(0,stopInd - stepWindow)
        valMin=min(valOpt[startInd:stopInd])
        #if not isinstance(valOpt, float):
        #    stopInd = len(valOpt)
        #    startInd=max(0,stopInd - stepWindow)
        #    valMin=min(valOpt[startInd:stopInd])
        #else:
        #    stopInd = 1
        #    valMin = valOpt
        #   passed = (~adaptStep) || (step <= stepMin) || isempty(valMin) || (val >= valMin);


        passed=(it == 1) or (not adaptStep) or (step <= stepMin) or (val >= valMin)
        # if (passed):
        #     print("passed!")
        # else:
        #     print("not passed!")
        if not passed:
            print("not passed!")


        # Save the stepsize and pass/fail result if history requested
        if itSaveHist:
            estHist.step[itSaveHist]=step
            estHist.stepMax[itSaveHist]=stepMax
            estHist.passed[itSaveHist]=passed
        # If passed, set the optimal values and compute a new target sHat and snew.
        if (passed):
            # Save states that "passeded"
            A2xVarOpt=A2xVar
            pVarOpt=pVar
            sHatOpt=sHat
            sVarOpt=sVar
            rVarOpt=rVar
            xHatDampOpt=xHatDamp
            xHatOpt=xHat
            # Save record of "passed" utilities
            if (compVal):
                valOpt.append(val)
                #valOpt=np.concatenate(valOpt,val)

            # Store variables for export
            pHatFinal=pHat
            pVarFinal=pVar
            zHatFinal=zHat
            zVarFinal=zVar
            xHatPrevFinal=xHatFinal     # previous xHat
            xHatFinal=xHat
            xVarFinal=xVar
            rHatFinal=rHat
            rVarFinal=rVarOpt*scaleFac  # report unscaled version
            AxHatFinal=AxHat
            sHatFinal=sHatOpt / scaleFac    # report unscaled version
            sVarFinal=sVarOpt / scaleFac    # report unscaled version


            ## Check for convergence
            if (it > 1) and (stop == False):

                if (norm(xHatPrevFinal  -  xHatFinal) / norm(xHatFinal) < tol):
                    stop=True
                else:
                    if customStop == 1:
                        stop=stopFcn[val,xHatFinal,xHatPrevFinal,AxHatFinal]
                    else:
                        if customStop == 2:
                            S=struct('it',it,'val',val,'xHatPrev',xHatPrevFinal,'AxHat',AxHatFinal,'xHat',xHatFinal,'xVar',xVarFinal,'rHat',rHatFinal,'rVar',rVarFinal,'pHat',pHatFinal,'pVar',pVarFinal,'zHat',zHatFinal,'zVar',zVarFinal,'sHat',sHatFinal,'sVar',sVarFinal)
                            stop=stopFcn2[S]

            ## Set scaleFac to mean of pVar if variance-normalization is on.
            # Else scaleFac remains at the initialized value of 1 and has no effect
            if varNorm:
                scaleFac=mean(pVarRobust)

            ####################################################################
            # 3. Output nonlinear stage: find sHat and sVar (MAP estimate of Z using the "prior" and Y)
            # print("3. Output nonlinear stage")
            zHat,zVar=scaEstOut.estim(pHat,pVarRobust,nargout=2)

            # # DEBUGGING ---------
            # printMat(zHat, 'zHat')
            # printMat(zVar, 'zVar')
            # # --------------------

            sHatNew=np.multiply(scaleFac / pVarRobust, zHat - pHat)
            sVarNew=np.multiply(scaleFac / pVarRobust, 1.0 - np.minimum(zVar / pVarRobust,zVarTopVarMax))

            # # DEBUGGING ---------
            # printMat(sHatNew, 'sHatNew')
            # printMat(sVarNew, 'sVarNew')
            # # --------------------


            if bbStep and it > 2:
                # Compute previous step-direction/size weighted with column norms
                sBB=(xHatOpt[:n,:] - xHatDampOpt[:n,:])
                # Select the smallest stepsize over all the columns for a matrix
            # valued signal
                values=sum(abs(multiply(sBB,columnNorms)) ** 2,1) / sum(abs(np.dot(A,sBB) ** 2),1)
                step=min(values)
            # Increase stepsize, keeping within bounds
            step=min( stepIncr*max(step,stepMin), stepMax )
        else:
            # Automatically decrease stepMax (when opt.maxBadSteps<Inf)
            failCount=failCount + 1
            if failCount > maxBadSteps:
                failCount=0
                stepMax=max(stepMin,dot(maxStepDecr,stepMax))
            # Decrease stepsize, keeping within bounds
            step=min(max(stepMin,dot(stepDecr,step)),stepMax)
            if step < stepTol:
                stop=True
        # Save results in history
        if (itSaveHist):
            estHist.pHat[:,itSaveHist]=ravel(pHatFinal)
            estHist.pVar[:,itSaveHist]=ravel(pVarFinal)
            estHist.zHat[:,itSaveHist]=ravel(zHatFinal)
            estHist.zVar[:,itSaveHist]=ravel(zVarFinal)
            estHist.sHat[:,itSaveHist]=ravel(sHatFinal)
            estHist.sVar[:,itSaveHist]=ravel(sVarFinal)
            estHist.rHat[:,itSaveHist]=ravel(rHatFinal)
            estHist.rVar[:,itSaveHist]=ravel(rVarFinal)
            estHist.xHat[:,itSaveHist]=ravel(xHatFinal)
            estHist.xVar[:,itSaveHist]=ravel(xVarFinal)
            estHist.AxHat[:,itSaveHist]=ravel(AxHatFinal)
            estHist.val[itSaveHist]=val
            estHist.scaleFac[itSaveHist]=scaleFac
        # Print results
        if (verbose):
            fprintf(1,'it=%3d  val=%12.4e  stepsize=%f  |dx|/|x|=%12.4e\\n',it,val,step,norm(ravel(xHatPrevFinal) - ravel(xHatFinal)) / norm(ravel(xHatFinal)))
        # Apply damping to sHat, sVar, and xHat
        if (it == 1):
            if any(isnan(sVarOpt)):
                sVarOpt=sVarNew
            if any(isnan(xHatDampOpt)):
                xHatDampOpt=xHatOpt
        sHat=(1 - step)*sHatOpt + step*sHatNew
        sVar=(1 - step)*sVarOpt + step*sVarNew
        sVar[abs(sVar) < eps]=eps
        xHatDamp=(1 - step)*xHatDampOpt + step*xHatOpt


        ########################################################################
        ## 4. Input linear stage: find rHat and rVar (noisy measurement of X)
        # print("4. Input linear stage")
        # Step in rVar
        #rVar=1.0 / A.multSqTr(sVar)
        rVar=1.0 / (np.square(A).transpose()*sVar)  # Eq. 11a % rVar = 1./((A.^2)*sVar)

        if rVarStep:
            if (it == 1):
                if any(isnan(rVarOpt)):
                    rVarOpt=copy(rVar)
            rVar=dot((1 - step),rVarOpt) + dot(step,rVar)

        #  update rHat
        #rHat=xHatDamp + rVar*(A.multTr(sHat))
        rHat=xHatDamp + np.multiply(rVar, np.dot(A.transpose(),sHat))
        rVarRobust=np.maximum(rVar,rVarMin)

        # ### [DEBUG]
        # printMat(rHat,'rHat')
        # printMat(rVarRobust,'rVarRobust')
        # ###

        ########################################################################
        ## 5. Input nonlinear stage: find xHat and xVar (update X)
        # update xHat and xVar  using g_in function
        if compVal:
            # Send messages to input estimation function.
            if MsgUpdateFlag:
                valMsg=scaEstIn.msgUpdate(it,rHat,rVarRobust)
            else:
                valMsg=0

            # Compute mean, variance, and negative KL-divergence
            xHat,xVar,valIn=scaEstIn.estim(rHat, rVarRobust*scaleFac, nargout=3)
            valIn=sum(valIn) + valMsg
        else:
            # Compute mean and variance
            xHat,xVar=scaEstIn.estim(rHat, rVarRobust*scaleFac, nargout=2)

        # ### [DEBUG]
        # printMat(xHat,'xHat')
        # printMat(xVar,'xVar')
        # ###

    # end of main loop
    ############################################################################

    # Store "next" (i.e., post-"final") estimates for export
    xHatNext=xHat
    xVarNext=xVar
    sHatNext=sHat / scaleFac
    sVarNext=sVar / scaleFac
    # Trim the history if early termination occurred
    if saveHist:
        nitTrim=arange(floor(it / histIntvl))
        if (it < nit):
            estHist.xHat = estHist.xHat(arange(),nitTrim)
            estHist.xVar = estHist.xVar(arange(),nitTrim)
            estHist.AxHat = estHist.AxHat(arange(),nitTrim)
            estHist.pHat = estHist.pHat(arange(),nitTrim)
            estHist.pVar = estHist.pVar(arange(),nitTrim)
            estHist.zHat = estHist.zHat(arange(),nitTrim)
            estHist.zVar = estHist.zVar(arange(),nitTrim)
            estHist.sHat = estHist.sHat(arange(),nitTrim)
            estHist.sVar = estHist.sVar(arange(),nitTrim)
            estHist.rHat = estHist.rHat(arange(),nitTrim)
            estHist.rVar = estHist.rVar(arange(),nitTrim)
            estHist.passed = estHist.passed(nitTrim)
            estHist.val = estHist.val(nitTrim)
            estHist.scaleFac = estHist.scaleFac(nitTrim)
            estHist.step = estHist.step(nitTrim)
            estHist.stepMax = estHist.stepMax(nitTrim)
        estHist.it = nitTrim*histIntvl

    # Trim the outputs if mean removal was turned on
    if (opt.removeMean):
        xHatNext=A.contract(xHatNext)
        xVarNext=A.contract(xVarNext)
        xHatDamp=A.contract(xHatDamp)
        xHatFinal=A.contract(xHatFinal)
        xVarFinal=A.contract(xVarFinal)
        xHatPrevFinal=A.contract(xHatPrevFinal)
        AxHatFinal=A.contract(AxHatFinal)
        pHatFinal=A.contract(pHatFinal)
        pVarFinal=A.contract(pVarFinal)
        pVarOpt=A.contract(pVarOpt)
        A2xVarOpt=A.contract(A2xVarOpt)
        zHatFinal=A.contract(zHatFinal)
        zVarFinal=A.contract(zVarFinal)
        sHatFinal=A.contract(sHatFinal)
        sVarFinal=A.contract(sVarFinal)
        sHatNext=A.contract(sHatNext)
        sVarNext=A.contract(sVarNext)
        rHatFinal=A.contract(rHatFinal)
        rVarFinal=A.contract(rVarFinal)

    if (saveHist):
        #return xHatFinal, opt, estHist
        return xHat, opt, estHist
    else:
        #return xHatFinal, opt
        return xHat, opt

    # Export outputs
    pass
    estFin.xHat = xHatFinal
    estFin.xVar = xVarFinal
    estFin.pHat = pHatFinal
    estFin.pVar = pVarFinal
    estFin.zHat = zHatFinal
    estFin.zVar = zVarFinal
    estFin.sHat = sHatFinal
    estFin.sVar = sVarFinal
    estFin.rHat = rHatFinal
    estFin.rVar = rVarFinal
    estFin.AxHat = AxHatFinal
    estFin.xHatPrev = xHatPrevFinal
    estFin.xHatNext = xHatNext
    estFin.xVarNext = xVarNext
    estFin.xHatDamp = xHatDamp
    estFin.pVarOpt = pVarOpt
    estFin.rVarOpt = rVarOpt
    estFin.A2xVarOpt = A2xVarOpt
    estFin.sHatNext = sHatNext
    estFin.sVarNext = sVarNext
    estFin.val = val
    estFin.valIn = valIn
    if isscalar(valOpt):
        estFin.valOpt = [valOpt]
    else:
        estFin.valOpt = valOpt
    estFin.scaleFac = scaleFac
    estFin.step = step
    estFin.stepMax = stepMax
    estFin.failCount = failCount
    estFin.nit = it
    out1=estFin
    out2=opt
    if (saveHist):
        out3=estHist