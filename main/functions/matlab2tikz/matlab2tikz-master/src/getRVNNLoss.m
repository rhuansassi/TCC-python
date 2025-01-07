function y = getRVNNLoss(x, numEpochs)
    
  subIndex = 0;
  count = 1;
  while (subIndex == 0)
    if (rem((length(x) - count), numEpochs) == 0)
        subIndex = count + 1;
    end
    count = count + 1;
  end
  
  xFix = x(subIndex:end);
  step = length(xFix)/numEpochs;
  
  indexes = 1:step:length(xFix);
  
  for k=1:length(indexes)
     index = indexes(k);
      
        y(k) = mean(xFix(index:index+(step - 1)));
  end
end