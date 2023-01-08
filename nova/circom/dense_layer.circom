pragma circom 2.1.1;

template Main() {
    signal input a;
    signal input b; 

    signal input c; 

    a + b === c;
    
    log("- Constants Satisfied -");
}

component main { public [ a, b ] } = Main();
