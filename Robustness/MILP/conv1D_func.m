
function out = conv1D_func(input, weight, bias, padding, stride)

num_filter = size(weight,1);
stacked_kernel_size = size(weight,2)*size(weight,3);
% conv_weight = flip(weight,3);
kernel = reshape(permute(weight,[1 3 2]), [num_filter, stacked_kernel_size]);

k = size(weight,3);
p = padding;
s = stride;




input_stretched =  padarray(zeros(size(input)), [0,p],'both');
input_stretched = double2sdpvar(input_stretched);
input_stretched(:,1+p:end-p) = input;
% input =  padarray(input, [0,p],'both');

input_conv_window2 = [];

for i =1:size(input,1)
    if k<=1 
        input_conv_window2 = [input_conv_window2; im2col(input_stretched(i,:),[1 k])'];
    else
        input_conv_window2 = [input_conv_window2; im2col(input_stretched(i,:),[1 k])];
    end
    
end


out = kernel * input_conv_window2(:,1:s:end) + bias;
