function out = deconv1D_func(input, weight, bias, padding, stride)
num_filter = size(weight,2);
stacked_kernel_size = size(weight,1)*size(weight,3);

conv_weight = flip(weight,3);
kernel = reshape(permute(conv_weight,[2 3 1]), [num_filter, stacked_kernel_size]);

k = size(conv_weight,3);
p = padding;
s = stride;
p_prime = k-p-1;

if s>1
    input_stretched = zeros(size(input).*[1,s]-[0,1]);
    
    input_stretched =  padarray(input_stretched, [0,p_prime],'both');
    input_stretched = double2sdpvar(input_stretched);
    input_stretched(:,1+p_prime:s:end-p_prime) = input;
else
    input_stretched =  padarray(zeros(size(input)), [0,p_prime],'both');
    input_stretched = double2sdpvar(input_stretched);
    input_stretched(:,1+p_prime:s:end-p_prime) = input;
    
end
% input_stretched =  padarray(input_stretched, [0,p_prime],'both');
input_conv_window2 = [];

for i =1:size(input,1)
    input_conv_window2 = [input_conv_window2; im2col(input_stretched(i,:),[1 k])];
end


out = kernel * input_conv_window2 + repmat(bias,[1 300]);