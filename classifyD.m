function label = classifyD(a, x)
    inProd = (a.') * x;
    p0 = 1 / (1 + exp(inProd));
    p1 = 1 / (1 + exp(-inProd));
    p = [p0, p1];
    
    % label stores argmax (i.e. index of p that gave max)
    [~, label] = max(p);
    
    % first element corresponds to label = 0, second corresponds to
    % label = 1 (handle one-based indexing)
    label = label - 1;
end