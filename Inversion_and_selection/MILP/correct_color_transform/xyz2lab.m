function Lab = xyz2lab(XYZ,refWhite)
    % Copyright EPFL-LSP
    % Derived from version 2005 (R.D Hersch / F. Crété)
    % Wyszecki, p. 167
    % XYZ may be a n x 3 matrix containg in each row one XYZ value.
    % In that case n x 3 matrix will be returned containing
    % one Lab value for each XYZ value, EP, 21 3 03
	X = XYZ(:, 1); 	Y = XYZ(:, 2); 	Z = XYZ(:, 3);

    % ./ denotes element wise division
    xxn = X ./ refWhite(1);  yyn = Y ./ refWhite(2);  zzn = Z ./ refWhite(3);
    
    % See Lab_2_XYZ.m for explanation
    i1 = find(xxn >  0.008856); [~, i1y] = size(i1);
    i2 = find(xxn <= 0.008856); [~,i2y] = size(i2);
    fxxn = ones(i1y + i2y, 1);
	if i1y > 0
		fxxn1 = xxn(i1).^(1/3);
        fxxn(i1) = fxxn1;
    end
    if i2y > 0
		fxxn2 = 7.787 * xxn(i2) + 16/116;
        fxxn(i2) = fxxn2;
    end
  
    i1 = find(yyn >  0.008856); [~, i1y] = size(i1);
    i2 = find(yyn <= 0.008856); [~, i2y] = size(i2);
    fyyn = ones(i1y + i2y, 1);
    L = ones(i1y + i2y, 1);
	if i1y > 0
        fyyn1 = yyn(i1).^(1/3);
        L1 = 116 * fyyn1 - 16;
        fyyn(i1) = fyyn1; 
        L(i1) = L1; 
    end;
    if i2y > 0
        fyyn2 = 7.787 * yyn(i2) + 16/116;
        L2 = 903.3 * yyn(i2);
        fyyn(i2) = fyyn2;
        L(i2) = L2;
    end;
    
    i1 = find(zzn >  0.008856); [~, i1y] = size(i1);
    i2 = find(zzn <= 0.008856); [~, i2y] = size(i2);
    fzzn = ones(i1y + i2y, 1);
	if i1y > 0
        fzzn1 = zzn(i1).^(1/3);
        fzzn(i1) = fzzn1;
    end;
    if i2y > 0
        fzzn2 = 7.787 * zzn(i2) + 16/116;
        fzzn(i2) = fzzn2;
    end;

    a = 500 * (fxxn - fyyn);
    b = 200 * (fyyn - fzzn);
    
    Lab = [L, a, b];

