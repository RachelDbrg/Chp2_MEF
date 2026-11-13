//+
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Rectangle(2) = {0, 0, 0, 1, 1, 0};
//+
Physical Surface("Elements", 9) = {1};
//+
Physical Curve("gauche", 10) = {4};
//+
Physical Curve("haut", 11) = {3};
//+
Physical Curve("droite", 12) = {2};
//+
Physical Curve("bas", 13) = {1};
