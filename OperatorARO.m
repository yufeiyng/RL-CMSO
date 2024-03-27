function Offspring = OperatorARO(Problem, Parent, N)

    if isa(Parent(1), 'SOLUTION')
        evaluated = true;
        Parent = Parent.decs;
    else
        evaluated = false;
    end

    D = Problem.D;
    Offspring = zeros(N, D);

    Direct1 = zeros(N, D);
    Direct2 = zeros(N, D);
    theta = 2 * (1 - Problem.FE / Problem.maxFE);

    for i = 1:N
        L = (exp(1) - exp(((Problem.FE - 1) / Problem.maxFE) ^ 2)) * (sin(2 * pi * rand));
        rd = ceil(rand * D);
        Direct1(i, randperm(D, rd)) = 1;
        c = Direct1(i, :);

        R = L .* c;
        A = 2 * log(1 / rand) * theta;

        if A > 1
            K = [1:i - 1 i + 1:N];
            randIndex = randi(N - 1);
            Offspring(i, :) = Parent(K(randIndex), :) + R .* (Parent(i, :) - Parent(K(randIndex), :)) + round(0.5 * (0.05 + rand)) * randn;

        else
            Direct2(i, ceil(rand * D)) = 1;
            gr = Direct2(i, :);
            r4 = rand;
            H = ((Problem.maxFE - Problem.FE + 1) / Problem.maxFE) * r4;
            b = Parent(i, :) + H * gr .* Parent(i, :);
            Offspring(i, :) = Parent(i, :) + R .* (rand .* b - Parent(i, :));
        end

    end

    %% Polynomial mutation
    Lower = repmat(Problem.lower, N, 1);
    Upper = repmat(Problem.upper, N, 1);
    disM = 20;
    Site = rand(N, D) < 1 / D;
    mu = rand(N, D);
    temp = Site & mu <= 0.5;
    Offspring = max(min(Offspring, Upper), Lower);
    Offspring(temp) = Offspring(temp) + (Upper(temp) - Lower(temp)) .* ((2 .* mu(temp) + (1 - 2 .* mu(temp)) .* ...
        (1 - (Offspring(temp) - Lower(temp)) ./ (Upper(temp) - Lower(temp))) .^ (disM + 1)) .^ (1 / (disM + 1)) - 1);
    temp = Site & mu > 0.5;
    Offspring(temp) = Offspring(temp) + (Upper(temp) - Lower(temp)) .* (1 - (2 .* (1 - mu(temp)) + 2 .* (mu(temp) - 0.5) .* ...
        (1 - (Upper(temp) - Offspring(temp)) ./ (Upper(temp) - Lower(temp))) .^ (disM + 1)) .^ (1 / (disM + 1)));

    if evaluated
        Offspring = Problem.Evaluation(Offspring);
    end

end
