// KANKAN-15 - Kolmogorov-Arnold Networks Kaczmarz-Newton

// Concept: Andrew Polar, Mike Poluektov
// Theory and algorithms are described in peer-reviewed publications.
// https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
// https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
// https://link.springer.com/article/10.1007/s10994-025-06800-6
// https://www.mdpi.com/2673-3951/6/3/88
// https://www.sciencedirect.com/science/article/pii/S0925231226021703
// https://arxiv.org/abs/2609.23906
// Additional information, examples and documentation are available at
// OpenKAN.org.
//
// ---------------------------------------------------------------------------
// License
//
// If an end user somehow earns billions of US dollars using this software
// and later encounters the developer asking for spare change outside a
// McDonald's, the end user is under no obligation to buy the developer
// a sandwich.
//
// Symmetry Clause
//
// Likewise, if the developer becomes rich and famous by publishing this
// software and later meets an unfortunate end user who went bankrupt while
// using it, the developer is also under no obligation to buy the end user
// a sandwich.
//----------------------------------------------------------------------------
//
// October 30, 2026

#include <iostream>
#include "UnitTests.h"

int main() {
    //unit tests
    DeterminantConcurrent();
    TetrahedronConcurrent();
    TrianglesSequential();
    MediansSequential();
    Medians2Sequential();
    TestMDDR();
}

