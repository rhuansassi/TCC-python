clc; clear; close all;

%% GFDM Channel Estimation Comparison Process
% Deep Learning techniques: SCF & RVNN
% Classical CE techniques: LS & LMMMSE 
% Marcos Nascimento, 2024


%% Simulation Parameters
cpLenghts = [16];               % Simulated CP sizes by modulation orders
MOrders = [4];                  % Simulated modulation order (M-QAM)

applyNonLinear = false;         % Apply NonLinearities (Clipping noise)
numSymbolsComparison = 1000;    % Number of GFDM symbols to be compared each interation
numIterationsPerSNR = 11;       % Number of iterations for each SNR simulated

%% Environment Loading

disp('Setup Environment...')
addpath('functions');
addpath('functions/gfdmlib');
addpath('functions/scripts');
addpath('Functions/neural-networks/scf-nn/');

setPath; clc;
%% Parameters

% Errors Plot Controls
showTrainErrors = true;         % Show trainining Errors
showValidationErrors = false;   % Show validation Errors

% Training Parameters
doTraining = true;              % If false skip training step
numSymbols = 5000;              % Number of GFDM Blocks for training set
epochs = 70;                    % Training Epochs
validFreq = 10;                 % Validation Frequency (must be multiples of epochs)


simulationCount = 1;
tic;
for MIndex = 1:length(MOrders)
    modOrder = MOrders(MIndex); 
    for cpLengthIndex = 1:length(cpLenghts)
        cpLength = cpLenghts(cpLengthIndex);
        
        % GFDM Signal
        numSubCarrier = 64;
        numSubSymbols = 3; 
        numPilotSubSymbols = 1;
        pilotFreq = getPilotFrequency(modOrder);
        
        % Modulation
        modType = 'QAM';
        EbNOdB = getEbN0(modOrder);
        
        %% Setup GFDM LIB instance for DL methods
        % (Check GFDMLib - Vodafone examples for configuration examples) 
        pDL = get_defaultGFDM('BER');
        pDL.modType = modType; 
        pDL.K = numSubCarrier;
        pDL.Mp = 0;
        pDL.Md = numSubSymbols-pDL.Mp;
        pDL.M = pDL.Mp+pDL.Md;
        pDL.Kon = numSubCarrier;
        pDL.Ncp = cpLength;
        pDL.mu = log2(modOrder);
        pDL.deltaK = 0;
        pDL.applyNonLinearities = applyNonLinear;
        
        %% Setup GFDM LIB instance for classical CE methods
        % (Check GFDMLib - Vodafone examples for configuration examples) 
        pCE = get_defaultGFDM('BER');
        pCE.modType = modType; 
        pCE.K = numSubCarrier;
        pCE.Mp = numPilotSubSymbols;
        pCE.Md = numSubSymbols-pCE.Mp;
        pCE.M = pCE.Mp+pCE.Md;
        pCE.Kon = numSubCarrier;
        pCE.Ncp = cpLength;
        pCE.mu = log2(modOrder);
        pCE.deltaK = pilotFreq;
        pCE.applyNonLinearities = applyNonLinear;
        
        %% Channel Model
        SNR_dB = getRealValuedSNR(modOrder);    
        [channel, h, pathPower, delaySpread] = generateChannelModel(pDL);

        %% Data generation
        disp('Generating Real Valued Data...')
        [XTrain, YTrain, XValid, YValid, dataTrain, dataValid] = generateDataSet(pDL, numSymbols, channel, SNR_dB, h, 'none');
        NInputs = size(XTrain,1);
        
        %% Real Valued Neural Network Structure
        scalingFactorRVNN = sqrt(NInputs);                      % Dataset Normalization factor
        
        % Data Structuring for RealValued - NN
        % Splits real and imaginary part to apply on RV-DL model
        [XTrainStruct, YTrainStruct, ... 
         XValidStruct, YValidStruct] = processInputNN(XTrain/scalingFactorRVNN, ...
                                                      YTrain/scalingFactorRVNN, ...
                                                      XValid/scalingFactorRVNN, ...
                                                      YValid/scalingFactorRVNN, ...
                                                      false);  
        % Input Layer dimension
        inputLayerDim = 2*size(XTrain, 1);     

        % Number of Neurons per layer
        numNeuronsPerLayer = getNumNeuronsPerLayer(pDL);        
        
        % Network structure
        lgraph = [...
            featureInputLayer(inputLayerDim,'Name','input')
        
            fullyConnectedLayer(numNeuronsPerLayer,'Name','linear1')
            tanhLayer('Name','tanh1')
            
            fullyConnectedLayer(inputLayerDim,'Name','linearOutput')
            regressionLayer('Name','output')];
        
        % Number of epochs and Its/epoch.
        maxEpochs = 200;
        iterPerEpoch = 512;
        
        options = trainingOptions('adam', ...
            MaxEpochs=maxEpochs, ...
            InitialLearnRate=1e-4, ...
            LearnRateDropFactor=0.98, ...
            LearnRateDropPeriod=5, ...
            LearnRateSchedule='piecewise', ...
            Shuffle='every-epoch', ...
            ValidationData={XValidStruct.',YValidStruct.'}, ...
            ValidationFrequency=iterPerEpoch, ...
            ValidationPatience=5, ...
            ExecutionEnvironment='cpu', ...
            Plots='training-progress', ...
            Verbose=false);

        %% SCFNN Parameters

        disp('Initializing SCFNN...');

        K         = size(XTrain,1);        % Number of Inputs
        M         = K;                     % Number of Outputs
        
        % No hidden layer structure
        N          = [K M];                % Number of Neurons per layer MLP
        L          = length(N)-1;          % Number of layers without input
        act        = ["th"];               % Activation functions
        eta        = [0.01];               % Adaptive steps of weights and bias

        % 1 Hidden layer structure
%         N         = [K 256 M];           % Number of Neurons per layer MLP
%         L         = length(N)-1;         % Number of layers without input
%         act       = ["th" "th"];         % Activation functions
%         eta       = [0.01 0.01];         % Adaptive steps of weights and bias
        

        % Init SCFNN instance
        netSCF = createSCFNN(N,act,eta,[],[],[]);
        
        % Normalization 
        scalingFactorSCF = sqrt(NInputs);  % Dataset Normalization factor
        XTrainNorm_SCF = XTrain/scalingFactorSCF;
        YTrainNorm_SCF = YTrain/scalingFactorSCF;
        XValidNorm_SCF = XValid/scalingFactorSCF;
        YValidNorm_SCF = YValid/scalingFactorSCF;
        
        
        %% Training Process
        
        % Variables initializing
        [~, nBlocksTrain] = size(XTrain);
        [~, nBlocksVal] = size(XValid);
        mseTrain_SCF = zeros(nBlocksTrain, epochs);
        mseVal_SCF = zeros(nBlocksVal, epochs/validFreq);
             
        if doTraining 

            % Training RVNN
            disp('Training RVNN')
            [trainedNetRVNN, tr] = trainNetwork(XTrainStruct.',YTrainStruct.',lgraph,options);
        
            % Training CVNN
            for epoch = 1:epochs
                isValidationEpoch = rem(epoch,validFreq) == 0;
                
                % Data shuffle each epoch
                shuffle = randperm(length(XTrainNorm_SCF));
        
                % Training iteration
                for k = 1:nBlocksTrain

                    % Input/Output selection
                    idx = shuffle(k);
                    input_SCF = XTrainNorm_SCF(:,idx);
                    target_SCF = YTrainNorm_SCF(:,idx);
        
                    % Training Iteration
                    netSCF = trainSCFNN(netSCF, input_SCF, target_SCF, epoch);
                    
                    % Training MSE
                    mseTrain_SCF(k,epoch) = mseTrain_SCF(k,epoch) + mean(abs(target_SCF-netSCF.y).^2);
                end
        
                if showTrainErrors 
                    disp('=================================================')
                    disp('Epoch: ')
                    disp(epoch)
                    disp('Training Errors [dB]:')
                    disp(' ');
                    disp('     SCF   ')
                    disp(['   ', num2str(10*log10(mean(mseTrain_SCF(:,epoch))))]);
                    disp('=================================================')
                end
                    
                % Validation Inference
                if isValidationEpoch
                    for j=1:nBlocksVal   
        
                        % Input/Output selection
                        input_SCF = XValidNorm_SCF(:,j);
                        target_SCF = YValidNorm_SCF(:,j);
        
                        % Inference
                        netSCF = inferenceSCFNN(netSCF, input_SCF);
        
                        % Validation MSE
                        mseVal_SCF(j,epoch/validFreq) = mseVal_SCF(j,epoch/validFreq) + mean(abs(target_SCF-netSCF.y).^2);
                    end
        
                    if showValidationErrors
                        disp('=================================================')
                        disp('Epoch: ')
                        disp(epoch)
                        disp('Validation Errors [dB]')
                        disp('[ SCF ] ')
                        disp([num2str(10*log10(mean(mseVal_SCF(:,epoch/validFreq))))]);
                        disp('=================================================')
                    end
                end
            end

            % Save training results
            if (applyNonLinear)
                linearText = 'NonLinear'
            else
                linearText = 'Linear';
            end
            
            convergenceName = sprintf('Conv_%s_%dQAM_%dCp.mat', linearText, 2^pDL.mu, pDL.Ncp);
            filePath = sprintf('./results/convergence/%s', convergenceName);
            
            save('./trained-net/trainedSCF.mat', 'netSCF', 'scalingFactorSCF');
            save('./trained-net/trainedNetRVNN.mat', 'trainedNetRVNN', 'tr', 'scalingFactorRVNN');
            save(filePath, 'mseTrain_SCF', 'mseVal_SCF', 'scalingFactorSCF', 'tr');
        end
        
        %% Parameters Techniques Comparison
        
        disp('Initializing Comparison process...')
        
        % Load TrainedNets
        load('./trained-net/trainedSCF.mat')
        load('./trained-net/trainedNetRVNN.mat')
        
        
        % Initialize DFT and Channel Autocorrelation Matrix for LMMSE Calculation
        N = pCE.K*pCE.M;
        freqPilotPositions = 1:pCE.deltaK*pCE.M:N;
        F = dftmtx(N)/sqrt(N);
        pCE.Fp = F(freqPilotPositions,:);
        a = 10.^(pathPower/10);
        P =  a.'./(sum(a));
        P_freq = sqrt(N)*pCE.Fp*diag([sqrt(P); zeros(N-delaySpread,1)]);
        R_HH = P_freq*P_freq';
        Hhat = ones(pCE.K*pCE.M, numSymbolsComparison);
        
        %% Technique Comparison Process
        
        % Intializations
        LS_SER_It = zeros(length(EbNOdB), numIterationsPerSNR);
        LMMSE_SER_It = zeros(length(EbNOdB), numIterationsPerSNR);
        SCF_SER_It = zeros(length(EbNOdB), numIterationsPerSNR);
        RVNN_SER_It = zeros(length(EbNOdB), numIterationsPerSNR);
                      
        for snr=1:length(EbNOdB)
            for i=1:numIterationsPerSNR
                
                disp('=================================================')
                disp('           Technique Comparison Proccess         ')
                disp('SNR [dB]:')
                disp(EbNOdB(snr))
                disp('Iteration:')
                disp(i)
                disp('=================================================')
            
                % Data generation
                [s, d, y, yNoCp, Xp] = generate_TestData(pCE, numSymbolsComparison, channel, EbNOdB(snr), h);
                [XTest, XLabels] = generateDataSetTest(pDL, numSymbolsComparison, channel, EbNOdB(snr), h, 'none');
                
                % Deep Learning Techniches Inference
                [scfOutput, rvNNOutput] = performInferenceNetworks(XTest, ...
                                                                  netSCF, scalingFactorSCF, ...
                                                                  trainedNetRVNN, scalingFactorRVNN);
                 
                % Classical Channel Equalization
                [dhat_LS, dhat_LMMSE] = ChannelEqualization(pCE, y, Xp, pCE.Fp, delaySpread, EbNOdB(snr), R_HH);
                
                % SER Comparison
                [LS_SER_It(snr,i), ...
                 LMMSE_SER_It(snr,i), ...
                 SCF_SER_It(snr,i),  ...
                 RVNN_SER_It(snr, i)] = get_SER_Comparison(pDL, pCE, s, XLabels, dhat_LS, dhat_LMMSE, scfOutput, rvNNOutput);
            end
        end
        
        % Average SER result 
        SCF_SER = sum(SCF_SER_It.')./(numIterationsPerSNR*N);
        RVNN_SER = sum(RVNN_SER_It.')./(numIterationsPerSNR*N);

        % discount pilots subcarriers for classical CE - SER
        numPilots = pCE.K/pCE.deltaK;
        LS_SER = sum(LS_SER_It.')./(numIterationsPerSNR*(N-numPilots));
        LMMSE_SER = sum(LMMSE_SER_It.')./(numIterationsPerSNR*(N-numPilots));

        % Channel Estimation Plots 
        figure(simulationCount);
        semilogy(EbNOdB,LMMSE_SER,'Color','#eeca10','LineWidth',2);
        hold on;
        semilogy(EbNOdB,LS_SER,'Color','#18ab2b','LineWidth',2);
        hold on;
        semilogy(EbNOdB,SCF_SER,'Color','#429abb','LineWidth',2);
        hold on;
        semilogy(EbNOdB,RVNN_SER,'Color','#9142bb','LineWidth',2);
        hold on;
        legend('LS', 'LMMSE', 'SCFNN', 'RVNN')
        xlabel('SNR [dB]'); ylabel('SER');
        title(sprintf('SER Comparison - %d QAM | %d CP', 2^pDL.mu, pDL.Ncp));
        grid on;
        
        % Save
        if (applyNonLinear)
            linearText = 'NonLinear'
        else
            linearText = 'Linear';
        end
        
        fileName = sprintf('%s_%dQAM_%dCp.mat', linearText, 2^pDL.mu, pDL.Ncp);
        filePath = sprintf('./results/convergence/%s', fileName);
        save(filePath, 'EbNOdB', 'LS_SER', 'LMMSE_SER', 'SCF_SER', 'RVNN_SER', 'LS_rxSymbols', 'LMMSE_rxSymbols', 'SCF_rxSymbols', 'RVNN_rxSymbols', 'pDL');
        
        simulationCount = simulationCount + 1;
    end
end
%%
toc;