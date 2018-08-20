function varargout = ode_pp(dowhat,params,varargin)

% Lotka-Volterra PP Model
% dxdt = alpha*x - beta*x*y
% dydt = delta*x*y - gamma*y

switch dowhat
    
    case 'setup'
        
        settings.finalt = 50;
        settings.deltat = 1/100;
        settings.timespan = 0:settings.deltat:settings.finalt;
        settings.u0 = [100;10];
        
        varargout{1} = settings;
        
    case 'solve'
        
        settings = varargin{1};
        
        alpha = params(1);
        beta = params(2);
        delta = params(3);
        gamma = params(4);
        
        dt= settings.deltat;
        N = settings.finalt/dt;
        u = settings.u0;
        sol = zeros(2,N+1);
        sol(:,1) = u;
        for k=1:N
            y1 = u;
            k1 = [alpha*y1(1) - beta*y1(1)*y1(2);delta*y1(1)*y1(2)-gamma*y1(2)];
            y2 = u+dt*k1/2;
            k2 = [alpha*y2(1) - beta*y2(1)*y2(2);delta*y2(1)*y2(2)-gamma*y2(2)];
            y3 = u+dt*k2/2;
            k3 = [alpha*y3(1) - beta*y3(1)*y3(2);delta*y3(1)*y3(2)-gamma*y3(2)];
            y4 = u+dt*k3;
            k4 = [alpha*y4(1) - beta*y4(1)*y4(2);delta*y4(1)*y4(2)-gamma*y4(2)];
            
            u = u+dt/6*(k1+2*k2+2*k3+k4);
            sol(:,k+1) = u;
        end
        
        varargout{1} = sol;
        
%         out = sol;
        
%         ct = deltat;
%         asol = zeros(3,length(settings.timespan));
%         asol(:,end) = settings.phiT;
%         prog = 1;
%         while ct < settings.finalt
%             ct = ct + deltat;
%             csol = 1/2*(sol(:,end-prog+1)+sol(:,end-prog));
%             pasol = asol(:,end-prog+1);
%             if prog > 1
%                 ppasol = asol(:,end-prog+1);
%             else
%                 ppasol = asol(:,end-prog+1);
%             end
%             pasol = 4/3*pasol - 1/3*ppasol;
%             t1 = (r).*(csol);
%             t2 = ones(3,1) - alpha*csol;
%             adata = pasol*intwt;
%             Jt = eye(3,3)*intwt;
%             j1 = r.*t2;
%             J1 = [j1(1) 0 0;0 j1(2) 0;0 0 j1(3)];
%             J2 = (-[t1 t1 t1].*alpha);
%             J = Jt-J1'-J2';
%             asol(:,end-prog) = J\adata;
%             prog = prog + 1;
%         end
%         settings.currentsol(1).sol = sol;%(:);
%         settings.currentsol(2).sol = asol;%(:);
%         
%         eest = input_ode_pp('error_estimate',params,settings);

%         eest
%         
%         if nargout == 5
%             varargout{1} = settings;
%             varargout{2} = [sol(3,end);zeros(length(params),1)];
%             varargout{3} = sol(:);
%             varargout{4} = asol(:);
%             varargout{5} = eest;
%         elseif nargout == 1
%             varargout{1} = [sol;asol];
%         elseif nargout == 2
%             varargout{1} = [sol;asol];
%             varargout{2} = eest;
%         end
        
%     case 'error_estimate'
%         
%         settings = varargin{1};
% %         sol = varargin{2};
%         fsol = settings.currentsol(1).sol;
%         asol = settings.currentsol(2).sol;
%             
% %         fsol = reshape(fsol,3,length(fsol)/3);
% %         asol = reshape(asol,3,length(asol)/3);
%         r = params(1:3)';
% %         alpha = settings.alpha;
%         alpha = [1 params(4) params(5);params(6) 1 params(7);params(8:9) 1];
%                 
%         N = length(settings.timespan);
%         f1 = fsol(:,1:end-1);
%         f2 = fsol(:,2:end);
%         
%         qpoints = [0.04691007703067;0.2307653449471;0.50000000000000;0.76923465505284;0.95308992296933]; % [pts;ones]
%         qweights = [0.11846344252810;0.23931433524968;0.28444444444445;0.23931433524968;0.11846344252810]; % weights
%         
%         t1 = settings.timespan(1:end-1)';
%         t2 = settings.timespan(2:end)';
%         dt = t2-t1;
%         a1 = asol(:,1:end-1);
%         a2 = asol(:,2:end);
%             
%         I = 0;
%         
%         for j = 1:length(qpoints)
%             fhat = qpoints(j)*f1 + (1-qpoints(j))*f2;
%             
%             ahat = qpoints(j)*a1 + (1-qpoints(j))*a2;
%                         
%             r1 = (repmat(r,1,N-1)).*(fhat);
%             r2 = ones(3,N-1) - alpha*(fhat);
%             rr = r1.*r2;
%             for k = 1:N-1
%                 I = I - qweights(j)*sum(dt(k)*((f2(:,k)-f1(:,k))/dt(k) - rr(:,k)).*ahat(:,k));
%             end
%         end
%         varargout{1} = I;
        
end
