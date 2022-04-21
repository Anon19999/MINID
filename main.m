disp("1. Run Experiments")
disp("2. Generate Plots")
e = input("Please select the option:");

switch e
    case 1
    % disp("Please select the experiment:")
    disp("1. Nano-Photonics Inversion (Rounded)")
    disp("2. Nano-Photonics Integer-Constrained Inversion")
    disp("3. Material Selection")
    disp("4. Material Selection (Inversison)")
    disp("5. Robustness")
    disp("6. Softrobot")
    disp("7. Contoning Inversion (Rounded)")
    disp("8. Contoning Integer-Constrained Inversion")
    disp("9. MILP-NA Combination")
    disp("10. 4-Ink Spectral Separation")
    disp("11. 44-Ink Spectral Separation")
    e = input("Please enter the experiment number:");


    switch e
        case 1 % Nano-Photonics Inversion (Rounded)
            run("Nano-Photonics/MILP/bound_tightening_full_parallel.m")
            run("Nano-Photonics/MILP/rounded_inversion_milp.m")
            run("Nano-Photonics/plot_gen.m")
        case 2 % Nano-Photonics Integer-Constrained Inversion
            run("Nano-Photonics/MILP/bound_tightening_full_parallel.m")
            run("Nano-Photonics/MILP/NN_MILP.m")
            run("Nano-Photonics/plot_gen.m")
        case 3 % Material Selection
            run("Inversion_and_selection/MILP/bound_tightening_full_parallel.m")
            run("Inversion_and_selection/MILP/NN_MILP.m")            
            run("Inversion_and_selection/MILP/map2image.m")
        case 4 % Material Selection (Inversison)
            run("Inversion_and_selection/MILP/bound_tightening_full_parallel.m")
            run("Inversion_and_selection/MILP/NN_MILP_parallel.m")
            run("Inversion_and_selection/MILP/map2image.m")
        case 5 % Robustness
            run("Robustness/MILP/bound_tightening_parallel_relu_lays.m")
            run("Robustness/MILP/bound_tightening_parallel.m")
            run("Robustness/MILP/NN_MILP.m")
            run("Robustness/MILP/sample_organizer.m")
            run("Robustness/MILP/plot_gen")
        case 6 %Softrobot
            run("Softrobot/MILP/bound_tightening_full_parallel.m")
            run("Softrobot/MILP/NN_MILP_parallel.m")
            run("Softrobot/MILP/plot_gen.m")
        case 7 %Contoning Inversion (Rounded)
            run("Contoning/MILP/bound_tightening_full_parallel.m")
            run("Contoning/MILP/rounded_inversion_MILP.m")
            run("Contoning/MILP/plot_gen.m")
        case 8 %Contoning Integer-Constrained Inversion
            run("Contoning/MILP/bound_tightening_full_parallel.m")
            run("Contoning/MILP/NN_MILP_integer_optimal.m")
            run("Contoning/MILP/plot_gen.m")
        case 9 % MILP + NA Combination
            run("MILP_NA_combination/MILP/bound_tightening_full_parallel.m")
            run("MILP_NA_combination/MILP/NN_MILP.m")
            run("MILP_NA_combination/MILP/plot_gen.m")
        case 10 % 4-Ink Spectral Separation
            run("Spectral separation/4-ink-net/MILP/bound_tightening_full_parallel.m")
            run("Spectral separation/4-ink-net/MILP/NN_MILP_parallel.m")
            run("Spectral separation/4-ink-net/MILP/grayRampGen.m")            
        case 11 % 44-Ink Spectral Separation
            run("Spectral separation/44-ink-net/MILP/bound_tightening_full_parallel.m")
            run("Spectral separation/44-ink-net/MILP/NN_MILP_parallel.m")
            run("Spectral separation/44-ink-net/MILP/grayRampGen.m")
    end
case 2
    disp("1. Nano-Photonics Inversion")
    disp("2. Material Selection")
    disp("3. Robustness")
    disp("4. Softrobot")
    disp("5. Contoning Inversion")
    disp("6. MILP-NA Combination")
    disp("7. 4-Ink Spectral Separation")
    disp("8. 44-Ink Spectral Separation")
    e = input("Please enter the experiment number:");
    switch e
        case 1
            run("Nano-Photonics/plot_gen.m")
        case 2
            run("Inversion_and_selection/MILP/map2image.m")
        case 3
            run("Robustness/MILP/plot_gen.m")
        case 4
            run("Softrobot/MILP/plot_gen.m")
        case 5
            run("Contoning/MILP/plot_gen.m")
        case 6
            run("MILP_NA_combination/MILP/plot_gen.m")
        case 7
            run("Spectral separation/4-ink-net/MILP/grayRampGen.m")
        case 8
            run("Spectral separation/44-ink-net/MILP/grayRampGen.m")
    end
end