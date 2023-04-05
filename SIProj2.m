%% -----------------------------------------ASSIGNING VALUES----------------------------------------------------
clear; % Cleaning the previous workspace so that we don't have any conflicts
load('iddata-19.mat');

u_id=id.InputData; % Used in prediction part
y_id=id.OutputData; % Used in prediction part
u_val=val.InputData; % Used in creating our not-linear ARX, in prediction and simuation part
y_val=val.OutputData; % Used to compare our results, in prediciton and simulation part

phi_id=[]; phi_val=[]; MSE_check=[]; % Initialization for recursive vectors
N = length(y_id); % For legibility , it's the length of which we will create our ARX models

%% -----------------------------------------FINDING MSE OPTIM---------------------------------------------------
%In order to FIND the best MSE, we check for every m=1:3, na=1:4, nb=1:4,and nk=1:3 all the possible results
for m = 1:3
    for na=1:4
        for nb=1:4
            for nk=1:3
                for k=1:N
                    comb=Comb_inator(y_id,u_id,na,nb,nk,m,k); % Create the vector of combinations
                    if(k==1) % Because the first row is only zeros, we get rid of it manually
                        phi_id=comb;
                    else
                        % For the first max(na,nb) rows, the length of the vector is not constant
                        % We will increase it by adding zeros at the end of it
                        if (length(comb)>length(phi_id))
                            for i=length(phi_id)+1:length(comb)
                                phi_id(k-1,i)=0;
                            end
                        end
                        phi_id=[phi_id;comb]; % Create the identification matrix of combinations
                    end
                    comb=Comb_inator(y_val,u_val,na,nb,nk,m,k); % Create the vector of combinations
                    if(k==1) % Because the first row is only zeros, we get rid of it manually
                        phi_val=comb;
                    else
                        % For the first max(na,nb) rows, the length of the vector is not constant
                        % We will increase it by adding zeros at the end of it
                        if (length(comb)>length(phi_val))
                            for i=length(phi_val)+1:length(comb)
                                phi_val(k-1,i)=0;
                            end
                        end
                        phi_val=[phi_val;comb]; % Create the validation matrix of combinations
                    end
                end

                theta=linsolve(phi_id,y_id); % Theta is our approximator, calculated only using the id matrix
                y_hat=phi_val*theta; % Approximation for the prediciton part

                MSE = immse(y_val,y_hat); % Mean square value
                MSE_check = [MSE_check; m, na, nb, nk, MSE]; % Putting the MSE in a matrix to check the best fit          
            end
        end
    end
end

% Because we have max(m)=3, we check how many rows have the same m
no_m=1;
while(MSE_check(no_m,1)==1)
    no_m=no_m+1;
end
no_m=no_m-1;

% MSE for m=1
mse_min=10000000;
for i=1:no_m
    if MSE_check(i,5)<mse_min
        mse_min = MSE_check(i, 5);
        index_min(1) = i;
    end
end

% MSE for m=2
mse_min=10000000;
for i=no_m+1:2*no_m
    if MSE_check(i,5)<mse_min
        mse_min = MSE_check(i, 5);
        index_min(2) = i;
    end
end

% MSE for m=3
mse_min=10000000;
for i=2*no_m+1:3*no_m
    if MSE_check(i,5)<mse_min
        mse_min = MSE_check(i, 5);
        index_min(3) = i;
    end
end

for i=1:3
    MSE_optim(i,:)=MSE_check(index_min(i),:);
end
%We see that the best MSE is for m=3, na=2, nb=2, and nk=2

%% ----------------------------------------------CLEANING--------------------------------------------------------
clear comb phi_id phi_val theta y_hat
% Cleaning the values of the matrices and the approximator theta so that we can make those for our optim m 

%% ---------------------------------------PREDICICTION FOR M OPTIM-----------------------------------------------
%Now that we have our optim values, we can proceed with the optim prediction
m=3; na=2; nb=2; nk=2; % Our optim values
for k=1:N
    comb=Comb_inator(y_id,u_id,na,nb,nk,m,k); % Create the vector of combinations
    if(k==1) % Because the first row is only zeros, we get rid of it manually
        phi_id=comb;
    else
        % For the first max(na,nb) rows, the length of the vector is not constant
        % We will increase it by adding zeros at the end of it
        if (length(comb)>length(phi_id))
            for i=length(phi_id)+1:length(comb)
                phi_id(k-1,i)=0;
            end
        end
        phi_id=[phi_id;comb]; % Create the identification matrix of combinations
    end
    comb=Comb_inator(y_val,u_val,na,nb,nk,m,k); % Create the vector of combinations
    if(k==1) % Because the first row is only zeros, we get rid of it manually
        phi_val=comb;
    else
        % For the first max(na,nb) rows, the length of the vector is not constant
        % We will increase it by adding zeros at the end of it
        if (length(comb)>length(phi_val))
            for i=length(phi_val)+1:length(comb)
                phi_val(k-1,i)=0;
            end
        end
        phi_val=[phi_val;comb]; % Create the validation matrix of combinations
    end
end

theta=linsolve(phi_id,y_id); % Theta is our approximator, calculated only using the id matrix
y_hat=phi_val*theta; % Approximation for the prediciton part

%% ------------------------------------------SIMULATION PART-----------------------------------------------------
% We need another y_hat for simulation that will change for every combination we make
% It needs to be initialised
y_hat_sim=zeros(1,N);

for k=1:N
    comb=Comb_inator(y_hat_sim,u_val,na,nb,nk,m,k); % Create the vector of combinations
    if(length(comb)<length(theta)) % We know the length at whitch comb should be, so we can use zeros function
        comb=[comb,zeros(1,length(theta)-length(comb))];
    end
    y_hat_sim(k)=comb*theta;

    % To keep track of our vector, we stack all the vectors in a matrix
    if(k==1)
        phi=comb;
    else
        phi=[phi;comb];
    end
end

%% ---------------------------------------------PLOTTING---------------------------------------------------------
% Plotting the prediction model output versus real output
figure;
plot(y_val);title('m_{optim} = 3');xlabel('Time');ylabel('Y');grid;
hold
plot(y_hat);legend('Y_{val}','Yhat');
% Plotting the simulation model output versus real output
figure
plot(y_val);title('m_{optim} = 3');xlabel('Time');ylabel('Y');grid;
hold;
plot(y_hat_sim);legend('Y_{val}','YhatSim');

%% ---------------------------------------------FUNCTION------------------------------------------------------------
% This function creates the vector of delayed inputs/outputs, then makes all the combination possible
% These are based around the polynomial degree m,
function combine = Comb_inator(y,u,na,nb,nk,m,k)
% Creating the vector of delayed inputs/outputs

for j=1:na
    if(k<=j)
        d(j)=0;
    else
        d(j)=y(k-j);
    end
end
for j=nk:nk+nb-1
    if(k<=j)
        d(j+na)=0;
    else
        d(j+na)=u(k-j);
    end
end

% Adjusting the vector so that we don't have null elements in between
%(This only applies for the first max(na,nb) rows)

if(k<=na)
    x=k;
    for i=k:na+nb
        if(i>na)
            d(x)=d(i);
            d(i)=0;
            x=x+1;
        end
    end
end

% Delay vector is not fully clear (we still have zeros after our non-null elements, and it affects it's length)
% We create an auxiliar vector, with only the non-null elements (to correct the length of our initial vector)

aux=zeros(1,1); % In case d is only made of zeros, we initialize aux
for i=1:na+nb
    if(d(i)~=0)
        aux(i)=d(i);
    end
end

% Now that we have a clear delay vector, we can make our combination based of the polynomial degree m

combine(1)=1; % For any m, because we only work with products, the first term must be 1
pnd=1; % Index so we can track where the next combination will take part

for i=1:length(aux) % If we used length(d), the actual length would have been higher because of the zeros
    combine(pnd)=combine(pnd)*aux(i); % Starts creating our combined terms using the aux vector
    % If the maximum degree is 1, increment the index and initialize the next term with 1
    if(m==1)
        pnd=pnd+1;
        combine(pnd)=1;
        % If not, move on
    else
        % In order not to lose terms with a lower polynomial degree than m, we keep the term and increment the index
        % We will also use that term for further combinations
        contor2=combine(pnd);
        combine(pnd+1)=contor2;
        pnd=pnd+1;
        for j=i:length(aux) % Index starts from position i so that we don't have any duplicates
            combine(pnd)=combine(pnd)*aux(j); % Starts creating our combined terms using the aux vector
            % If it the maximum degree is 2, increment the index and initialize the next term with contor2
            if(m==2)
                pnd=pnd+1;
                combine(pnd)=contor2;
                % If not, move on
            else
                % Same logic applies as above
                contor3=combine(pnd);
                combine(pnd+1)=contor3;
                pnd=pnd+1;
                for k=j:length(aux) % Index starts from position j so that we don't have any duplicates
                    combine(pnd)=combine(pnd)*aux(k); % Starts creating our combined terms using the aux vector
                    % If it the maximum degree is 2, increment the index and initialize the next term with contor3
                    if(m==3)
                        pnd=pnd+1;
                        combine(pnd)=contor3;
                    end
                end
                combine(pnd)=contor2; % Initialize the next terim with contor2 to be used in further combinations
            end
        end
        combine(pnd)=1; %Initialize the next terim with 1 to be used in further combinations
    end
end
% We can increase the maximum degree by adding additional lines of code like in the example above
% The logic is still the same, only the number of lines in our code change
end