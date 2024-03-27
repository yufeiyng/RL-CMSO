classdef RL_CMSO < ALGORITHM
    % <multi> <real/integer/label/binary/permutation> <constrained>

    methods

        function main(Algorithm, Problem)

            %% Parameter setting
            threshold = 1e-3; epsilon = 1e8; tao = 0.05;
            stage = 0; gen = 0; cnt = 0.1 * (Problem.maxFE / (2 * Problem.N));

            %% Generate random population
            Population1 = Problem.Initialization();
            Population2 = Population1;

            Fitness1 = CalFitness(Population1.objs, Population1.cons, 0);
            Fitness2 = CalFitness(Population2.objs, Population2.cons, epsilon);

            %% For QL
            numActions = 5; numPop = 4;
            alpha_ql = 0.01; gamma_ql = 0.9;
            greedy_ql = 0.1; c_ucb = 1;

            qlTable = QL_UCB(numPop, numActions, alpha_ql, gamma_ql, greedy_ql, c_ucb);
            clear alpha_ql gamma_ql greedy_ql c_ucb

            %% Optimization
            while Algorithm.NotTerminated(Population1)

                if mod(gen, cnt) == 0
                    action = qlTable.ChooseAction();
                    reward = zeros(numActions, numPop);
                end

                gen = gen + 1;

                if stage == 0
                    Objs2(gen) = sum(sum(Population2.objs, 1));
                    [FrontNo2, ~] = NDSort(Population2.objs, size(Population2.objs, 1));
                    NC2 = size(find(FrontNo2 == 1), 2);

                    state2 = IsStable(Objs2, gen, threshold);

                    if (state2 && NC2 == Problem.N) || Problem.FE > 0.5 * Problem.maxFE
                        CV2 = overall_cv(Population2.cons);
                        stage = 1; epsilon = max(CV2);
                        qlTable.Q = zeros(numActions, numPop);
                    end

                else
                    epsilon = (1 - tao) * epsilon;
                end

                N1 = ceil(Problem.N / action(1));
                N2 = ceil(Problem.N / action(2));
                N3 = ceil(Problem.N / action(3));
                N4 = ceil(Problem.N / action(4));

                Mat1 = TournamentSelection(2, N1, Fitness1);
                Mat2 = TournamentSelection(2, N2, Fitness2);
                Offspring3 = OperatorARO(Problem, Population1(Mat1).decs, N1);
                Offspring4 = OperatorARO(Problem, Population2(Mat2).decs, N2);

                Mat3 = TournamentSelection(2, N3, Fitness1);
                Mat4 = TournamentSelection(2, N4, Fitness2);
                [Offspring1, velocity1] = OperatorPSO1(Problem, Population1(Mat3));
                [Offspring2, velocity2] = OperatorPSO1(Problem, Population2(Mat4));

                if ~isempty(Offspring1)
                    Offspring1 = Deduplicate(Offspring1, [Population1.decs; Population2.decs], velocity1);
                end

                if ~isempty(Offspring1)
                    Offspring1 = Problem.Evaluation(Offspring1, velocity1);
                    Offspring2 = Deduplicate(Offspring2, [Population1.decs; Population2.decs; Offspring1.decs], velocity2);
                    Offspring3 = Deduplicate(Offspring3, [Population1.decs; Population2.decs; Offspring1.decs]);
                    Offspring4 = Deduplicate(Offspring4, [Population1.decs; Population2.decs; Offspring1.decs]);
                else
                    Offspring1 = [];
                end

                if ~isempty(Offspring2)
                    Offspring2 = Problem.Evaluation(Offspring2, velocity2);
                    Offspring3 = Deduplicate(Offspring3, Offspring2.decs);
                    Offspring4 = Deduplicate(Offspring4, Offspring2.decs);
                else
                    Offspring2 = [];
                end

                if ~isempty(Offspring3)
                    Offspring3 = Problem.Evaluation(Offspring3);
                    Offspring4 = Deduplicate(Offspring4, Offspring3.decs);
                else
                    Offspring3 = [];
                end

                if ~isempty(Offspring4)
                    Offspring4 = Problem.Evaluation(Offspring4);
                else
                    Offspring4 = [];
                end

                oldScore1 = mean(Fitness1);

                [Population1, Fitness1] = EnvironmentalSelection([Population1, Offspring1], Problem.N, true, 0);
                newScore1 = mean(Fitness1);
                reward(action(1), 1) = reward(action(1), 1) + (oldScore1 - newScore1);
                oldScore1 = newScore1;
                [Population1, Fitness1] = EnvironmentalSelection([Population1, Offspring3], Problem.N, true, 0);
                newScore1 = mean(Fitness1);
                reward(action(3), 3) = reward(action(3), 3) + (oldScore1 - newScore1);
                oldScore1 = newScore1;

                if stage == 0
                    oldScore2 = mean(CalFitness(Population2.objs, Population2.cons, epsilon));
                    [Population2, Fitness2] = EnvironmentalSelection([Population2, Offspring2], Problem.N, false, epsilon);
                    newScore2 = mean(Fitness2);
                    reward(action(2), 2) = reward(action(2), 2) + (oldScore2 - newScore2);
                    oldScore2 = newScore2;
                    [Population2, Fitness2] = EnvironmentalSelection([Population2, Offspring4], Problem.N, false, epsilon);
                    newScore2 = mean(Fitness2);
                    reward(action(4), 4) = reward(action(4), 4) + (oldScore2 - newScore2);
                else
                    [Population1, Fitness1] = EnvironmentalSelection([Population1, Offspring2], Problem.N, true, 0);
                    newScore1 = mean(Fitness1);
                    reward(action(2), 2) = reward(action(2), 2) + (oldScore1 - newScore1);
                    oldScore1 = newScore1;
                    [Population1, Fitness1] = EnvironmentalSelection([Population1, Offspring4], Problem.N, true, 0);
                    newScore1 = mean(Fitness1);
                    reward(action(4), 4) = reward(action(4), 4) + (oldScore1 - newScore1);
                end

                if stage == 0
                    [Population1, Fitness1] = EnvironmentalSelection([Population1, Offspring2, Offspring4], Problem.N, true, 0);
                else
                    [Population2, Fitness2] = EnvironmentalSelection([Population2, Offspring2, Offspring4], Problem.N, false, epsilon);
                end

                if mod(gen, cnt) == 0
                    qlTable = qlTable.UpdateQValue(action, reward);
                end

            end

        end

    end

end

function result = overall_cv(cv)
    cv(cv <= 0) = 0; cv = abs(cv);
    result = sum(cv, 2);
end

function result = IsStable(Objvalues, gen, threshold)
    result = 0;

    if gen ~= 1
        max_change = abs(Objvalues(gen) - Objvalues(gen - 1));

        if max_change <= threshold
            result = 1;
        end

    end

end
