clear; clc;

format long e;

set(groot, 'DefaultLineLineWidth', 1.5);

%parameters
N = 5000;
K = 20000;  % A: N*K
density = 0.01; % the proportion of nonzero elements in the sparse vector
MaxIter_outer_sca = 30; % maximum number of iterations
MaxIter_outer_gra = 100; % maximum number of iterations
MaxIter_outer_dec = 1000; % maximum number of iterations

Sample = 20; % number of repeatitions in the Monte Carlo simulations
plotSignal = 0; % plot the estimated signal if 1

% algDefaultSetting(algType, stepsizeType, numBlocks, MaxIter_outer, MaxIter_inner, Sample)
alg_descent1  = algDefaultSetting('dec', 'n/a',        1,  MaxIter_outer_dec, 1,  Sample); % "_descent" or "_dec" refers to the Bregman-based proximal gradient algorithm
alg_descent2  = algDefaultSetting('dec', 'n/a',        1,  MaxIter_outer_dec, 1,  Sample);

alg1_sca10_e  = algDefaultSetting('sca', 'exact',      1,  MaxIter_outer_sca, 10, Sample); % "_sca" refers to the partial linear approximation

alg2_sca1_e   = algDefaultSetting('sca', 'exact',      2,  MaxIter_outer_sca, 1,  Sample);
alg2_sca1_s   = algDefaultSetting('sca', 'successive', 2,  MaxIter_outer_sca, 1,  Sample);
alg2_sca10_e  = algDefaultSetting('sca', 'exact',      2,  MaxIter_outer_sca, 10, Sample);
alg2_sca10_s  = algDefaultSetting('sca', 'successive', 2,  MaxIter_outer_sca, 10, Sample);

alg2_gra_e    = algDefaultSetting('gra', 'exact',      2,  MaxIter_outer_gra, 1,  Sample); % "_gra" refers to the quadratic approximation
alg2_gra_s    = algDefaultSetting('gra', 'successive', 2,  MaxIter_outer_gra, 1,  Sample);

alg10_sca1_e  = algDefaultSetting('sca', 'exact',      10, MaxIter_outer_sca, 1,  Sample);
alg10_sca10_e = algDefaultSetting('sca', 'exact',      10, MaxIter_outer_sca, 10, Sample);

alg10_gra_e   = algDefaultSetting('gra', 'exact',      10, MaxIter_outer_gra, 1,  Sample);
alg10_gra_s   = algDefaultSetting('gra', 'successive', 10, MaxIter_outer_gra, 1,  Sample); 

for s = 1: 1: Sample
    disp(['Sample ' num2str(s) ' of ' num2str(Sample)]);
    
    % generating the parameters
    [A, y, mu, x_ori] = FUN_QI_parameter(N, K, density);
    
    % initialization
    x0 = randn(K,1);
%     x0 = x_ori;

    sol_descent1 = FUN_DescentLemma(A, y, mu, K, N, x0, alg_descent1, 1);
    alg_descent1.objval(s, :)  = sol_descent1.objval;
    alg_descent1.cputime(s, :) = sol_descent1.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol_descent1.x, alg_descent1.legend);
    end

    sol_descent2 = FUN_DescentLemma(A, y, mu, K, N, x0, alg_descent2, 10^-4);
    alg_descent2.objval(s, :)  = sol_descent2.objval;
    alg_descent2.cputime(s, :) = sol_descent2.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol_descent2.x, alg_descent2.legend);
    end
    
    % testing algorithms with different setups
    sol1_sca10_e = FUN_blockQI(A, y, mu, K, x0, alg1_sca10_e);
    alg1_sca10_e.objval(s, :)  = sol1_sca10_e.objval;
    alg1_sca10_e.cputime(s, :) = sol1_sca10_e.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol1_sca10_e.x, alg1_sca10_e.legend);
    end
    
    sol2_sca1_e = FUN_blockQI(A, y, mu, K, x0, alg2_sca1_e);
    alg2_sca1_e.objval(s, :)  = sol2_sca1_e.objval;
    alg2_sca1_e.cputime(s, :) = sol2_sca1_e.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol2_sca1_e.x, alg2_sca1_e.legend);
    end
    
    sol2_sca1_s = FUN_blockQI(A, y, mu, K, x0, alg2_sca1_s);
    alg2_sca1_s.objval(s, :)  = sol2_sca1_s.objval;
    alg2_sca1_s.cputime(s, :) = sol2_sca1_s.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol2_sca1_s.x, alg2_sca1_s.legend);
    end
    
    sol2_sca10_e = FUN_blockQI(A, y, mu, K, x0, alg2_sca10_e);
    alg2_sca10_e.objval(s, :)  = sol2_sca10_e.objval;
    alg2_sca10_e.cputime(s, :) = sol2_sca10_e.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol2_sca10_e.x, alg2_sca10_e.legend);
    end
    
    sol2_sca10_s = FUN_blockQI(A, y, mu, K, x0, alg2_sca10_s);
    alg2_sca10_s.objval(s, :)  = sol2_sca10_s.objval;    
    alg2_sca10_s.cputime(s, :) = sol2_sca10_s.cputime;
    if plotSignal
        plotEstimate(x_ori, sol2_sca10_s.x, alg2_sca10_s.legend);
    end

    sol2_gra_e = FUN_blockQI(A, y, mu, K, x0, alg2_gra_e);
    alg2_gra_e.objval(s, :)  = sol2_gra_e.objval;
    alg2_gra_e.cputime(s, :) = sol2_gra_e.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol2_gra_e.x, alg2_gra_e.legend);
    end

    sol2_gra_s = FUN_blockQI(A, y, mu, K, x0, alg2_gra_s);
    alg2_gra_s.objval(s, :)  = sol2_gra_s.objval;
    alg2_gra_s.cputime(s, :) = sol2_gra_s.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol2_gra_s.x, alg2_gra_s.legend);
    end
    
    sol10_sca1_e = FUN_blockQI(A, y, mu, K, x0, alg10_sca1_e);
    alg10_sca1_e.objval(s, :)  = sol10_sca1_e.objval;
    alg10_sca1_e.cputime(s, :) = sol10_sca1_e.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol10_sca1_e.x, alg10_sca1_e.legend);
    end
    
    sol10_sca10_e = FUN_blockQI(A, y, mu, K, x0, alg10_sca10_e);
    alg10_sca10_e.objval(s, :)  = sol10_sca10_e.objval;
    alg10_sca10_e.cputime(s, :) = sol10_sca10_e.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol10_sca10_e.x, alg10_sca10_e.legend);
    end
    
    sol10_gra_e = FUN_blockQI(A, y, mu, K, x0, alg10_gra_e);
    alg10_gra_e.objval(s, :)  = sol10_gra_e.objval;
    alg10_gra_e.cputime(s, :) = sol10_gra_e.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol10_gra_e.x, alg10_gra_e.legend);
    end

    sol10_gra_s = FUN_blockQI(A, y, mu, K, x0, alg10_gra_s);
    alg10_gra_s.objval(s, :)  = sol10_gra_s.objval;
    alg10_gra_s.cputime(s, :) = sol10_gra_s.cputime;    
    if plotSignal
        plotEstimate(x_ori, sol10_gra_s.x, alg10_gra_s.legend);
    end
    
    clear A x0 x_ori y mu
    
%    save('BlockQI_tmp');
end

figure;
hold on; box on; grid on;
semilogy(0: 1: alg1_sca10_e.MaxIter_outer,  mean(alg1_sca10_e.objval,1),  'DisplayName', alg1_sca10_e.legend)
semilogy(0: 1: alg2_sca1_e.MaxIter_outer,   mean(alg2_sca1_e.objval,1),   'DisplayName', alg2_sca1_e.legend)
semilogy(0: 1: alg2_sca1_s.MaxIter_outer,   mean(alg2_sca1_s.objval,1),   'DisplayName', alg2_sca1_s.legend)
semilogy(0: 1: alg2_sca10_e.MaxIter_outer,  mean(alg2_sca10_e.objval,1),  'DisplayName', alg2_sca10_e.legend)
semilogy(0: 1: alg2_sca10_s.MaxIter_outer,  mean(alg2_sca10_s.objval,1),  'DisplayName', alg2_sca10_s.legend)
semilogy(0: 1: alg2_gra_e.MaxIter_outer,    mean(alg2_gra_e.objval,1),    'DisplayName', alg2_gra_e.legend)
semilogy(0: 1: alg2_gra_s.MaxIter_outer,    mean(alg2_gra_s.objval,1),    'DisplayName', alg2_gra_s.legend)
semilogy(0: 1: alg10_sca1_e.MaxIter_outer,  mean(alg10_sca1_e.objval,1),  'DisplayName', alg10_sca1_e.legend)
semilogy(0: 1: alg10_sca10_e.MaxIter_outer, mean(alg10_sca10_e.objval,1), 'DisplayName', alg10_sca10_e.legend)
semilogy(0: 1: alg10_gra_e.MaxIter_outer,   mean(alg10_gra_e.objval,1),   'DisplayName', alg10_gra_e.legend)
semilogy(0: 1: alg10_gra_s.MaxIter_outer,   mean(alg10_gra_s.objval,1),   'DisplayName', alg10_gra_s.legend)
semilogy(0: 1: alg_descent1.MaxIter_outer,  mean(alg_descent1.objval,1),  'DisplayName', alg_descent1.legend)
semilogy(0: 1: alg_descent2.MaxIter_outer,  mean(alg_descent2.objval,1),  'DisplayName', alg_descent2.legend)
xlabel('number of iterations');
ylabel('objective value');
legend('show');
set(gca,'yscale','log')

figure;
hold on; box on; grid on;
semilogy(mean(alg1_sca10_e.cputime, 1), mean( alg1_sca10_e.objval,1), 'DisplayName', alg1_sca10_e.legend)
semilogy(mean(alg2_sca1_e.cputime,  1), mean( alg2_sca1_e.objval, 1), 'DisplayName', alg2_sca1_e.legend)
semilogy(mean(alg2_sca1_s.cputime,  1), mean( alg2_sca1_s.objval, 1), 'DisplayName', alg2_sca1_s.legend)
semilogy(mean(alg2_sca10_e.cputime, 1), mean( alg2_sca10_e.objval,1), 'DisplayName', alg2_sca10_e.legend)
semilogy(mean(alg2_sca10_s.cputime, 1), mean( alg2_sca10_s.objval,1), 'DisplayName', alg2_sca10_s.legend)
semilogy(mean(alg2_gra_e.cputime,   1), mean( alg2_gra_e.objval,  1), 'DisplayName', alg2_gra_e.legend)
semilogy(mean(alg2_gra_s.cputime,   1), mean( alg2_gra_s.objval,  1), 'DisplayName', alg2_gra_s.legend)
semilogy(mean(alg10_sca1_e.cputime, 1), mean(alg10_sca1_e.objval, 1), 'DisplayName', alg10_sca1_e.legend)
semilogy(mean(alg10_sca10_e.cputime,1), mean(alg10_sca10_e.objval,1), 'DisplayName', alg10_sca10_e.legend)
semilogy(mean(alg10_gra_e.cputime,  1), mean(alg10_gra_e.objval,  1), 'DisplayName', alg10_gra_e.legend)
semilogy(mean(alg10_gra_s.cputime,  1), mean(alg10_gra_s.objval,  1), 'DisplayName', alg10_gra_s.legend)
semilogy(mean(alg_descent1.cputime, 1), mean(alg_descent1.objval, 1), 'DisplayName', alg_descent1.legend)
semilogy(mean(alg_descent2.cputime, 1), mean(alg_descent2.objval, 1), 'DisplayName', alg_descent2.legend)
xlabel('CPU time');
ylabel('objective value');
legend('show');
set(gca,'yscale','log')

function plotEstimate(x_ori, x_estimate, algLegend)    
    figure; hold on;
    plot(x_ori, 'or');
    plot(x_estimate, 'xk');
    legend('original signal', 'estimated signal');
    xlabel('index'); ylabel('value');
    title(algLegend);
end

function algsetup = algDefaultSetting(algType, stepsizeType, numBlocks, MaxIter_outer, MaxIter_inner, Sample)    
    algsetup.MaxIter_inner = MaxIter_inner;
    algsetup.MaxIter_outer = MaxIter_outer;
    algsetup.algorithm = algType;
    algsetup.stepsize = stepsizeType;
    algsetup.P = numBlocks;
    if strcmp(algType, 'gra')
        algsetup.legend  = ['BGD: ' num2str(algsetup.P) ' block(s), stepsize ' algsetup.stepsize];
    elseif strcmp(algType, 'sca')
        algsetup.legend  = ['BSCA algorithm: ', num2str(algsetup.P) ' block(s), ' num2str(algsetup.MaxIter_inner) ' inner iterations, stepsize ' algsetup.stepsize];
    elseif strcmp(algType, 'dec')
        algsetup.legend  = ['Bregman-based proximal gradient algorithm'];
    end
        
    algsetup.objval  = zeros(Sample, MaxIter_outer + 1);
    algsetup.cputime = zeros(Sample, MaxIter_outer + 1);
end