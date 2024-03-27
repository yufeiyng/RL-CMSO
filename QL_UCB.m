classdef QL_UCB

    properties
        Q % Q table
        UCB % UCB table
        actionCounts % Action count table
        alpha % Learning rate
        gamma % Discount factor
        greedy % Epsilon-greedy parameter
        c % UCB exploration parameter
        numPop % Number of populations
        numActions % Number of actions
    end

    methods

        function obj = QL_UCB(numPop, numOperator, alpha, gamma, greedy, c)
            % Constructor
            obj.alpha = alpha;
            obj.gamma = gamma;
            obj.greedy = greedy;
            obj.c = c;
            obj.numPop = numPop;
            obj.numActions = numOperator;

            obj.Q = zeros(obj.numActions, obj.numPop);
            obj.UCB = zeros(obj.numActions, obj.numPop);
            obj.actionCounts = zeros(obj.numActions, obj.numPop);
        end

        function action = ChooseAction(obj)
            % Choose action using UCB exploration
            if rand < obj.greedy
                % Explore with epsilon-greedy
                action = randi(obj.numActions, [1, obj.numPop]);
            else
                % Exploit with UCB
                [~, action] = max(obj.UCB, [], 1);
            end

        end

        function obj = UpdateQValue(obj, action, reward)
            % Update Q-value using Q-learning
            for i = 1:obj.numPop
                obj.actionCounts(action(i), i) = obj.actionCounts(action(i), i) + 1;
                target = reward + obj.gamma * max(obj.Q, [], 1);
                obj.Q(action(i), i) = obj.Q(action(i), i) + obj.alpha * (target(action(i), i) - obj.Q(action(i), i));
                obj.UCB(:, i) = obj.Q(:, i) + obj.c * sqrt(log(sum(obj.actionCounts(:, i))) ./ (obj.actionCounts(:, i) + eps));
            end

        end

    end

end
