-- Based on https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua
local Seq2Seq = torch.class("neuralconvo.Seq2Seq")

function Seq2Seq:__init(vocabSize, hiddenSize)
  self.vocabSize = assert(vocabSize, "vocabSize required at arg #1")
  self.hiddenSize = assert(hiddenSize, "hiddenSize required at arg #2")

  self:buildModel()
end

function Seq2Seq:buildModel()
  self.encoder = nn.Sequential()
  self.encoder:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  self.encoder:add(nn.SplitTable(1, 2))
  self.encoderLSTM = nn.LSTM(self.hiddenSize, self.hiddenSize)
  self.encoder:add(nn.Sequencer(self.encoderLSTM))
  self.encoder:add(nn.SelectTable(-1))

  self.decoder = nn.Sequential()
  self.decoder:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  self.decoder:add(nn.SplitTable(1, 2))
  self.decoderLSTM = nn.LSTM(self.hiddenSize, self.hiddenSize)
  self.decoder:add(nn.Sequencer(self.decoderLSTM))
  self.decoder:add(nn.Sequencer(nn.Linear(self.hiddenSize, self.vocabSize)))
  self.decoder:add(nn.Sequencer(nn.LogSoftMax()))

  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()
end

function Seq2Seq:cuda()
  self.encoder:cuda()
  self.decoder:cuda()

  if self.criterion then
    self.criterion:cuda()
  end
end

function Seq2Seq:cl()
  self.encoder:cl()
  self.decoder:cl()

  if self.criterion then
    self.criterion:cl()
  end
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function Seq2Seq:forwardConnect(inputSeqLen)
  self.decoderLSTM.userPrevOutput =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevOutput, self.encoderLSTM.outputs[inputSeqLen])
  self.decoderLSTM.userPrevCell =
    nn.rnn.recursiveCopy(self.decoderLSTM.userPrevCell, self.encoderLSTM.cells[inputSeqLen])
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function Seq2Seq:backwardConnect()
  self.encoderLSTM.userNextGradCell =
    nn.rnn.recursiveCopy(self.encoderLSTM.userNextGradCell, self.decoderLSTM.userGradPrevCell)
  self.encoderLSTM.gradPrevOutput =
    nn.rnn.recursiveCopy(self.encoderLSTM.gradPrevOutput, self.decoderLSTM.userGradPrevOutput)
end

function Seq2Seq:train(input, target)
  local encoderInput = input
  local decoderInput = target:sub(1, -2)
  local decoderTarget = target:sub(2, -1)

  -- Forward pass
  local encoderOutput = self.encoder:forward(encoderInput)
  self:forwardConnect(encoderInput:size(1))
  local decoderOutput = self.decoder:forward(decoderInput)
  local Edecoder = self.criterion:forward(decoderOutput, decoderTarget)

  if Edecoder ~= Edecoder then -- Exist early on bad error
    return Edecoder
  end

  -- Backward pass
  local gEdec = self.criterion:backward(decoderOutput, decoderTarget)
  self.decoder:backward(decoderInput, gEdec)
  self:backwardConnect()
  self.encoder:backward(encoderInput, encoderOutput:zero())

  self.encoder:updateGradParameters(self.momentum)
  self.decoder:updateGradParameters(self.momentum)
  self.decoder:updateParameters(self.learningRate)
  self.encoder:updateParameters(self.learningRate)
  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()

  self.decoder:forget()
  self.encoder:forget()

  return Edecoder
end

local MAX_OUTPUT_SIZE = 20

function Seq2Seq:eval(input)
  assert(self.goToken, "No goToken specified")
  assert(self.eosToken, "No eosToken specified")

  self.encoder:forward(input)
  self:forwardConnect(input:size(1))

  local predictions = {}
  local probabilities = {}

  -- Forward <go> and all of it's output recursively back to the decoder
  local output = {self.goToken}

  for i = 1, MAX_OUTPUT_SIZE do
    local prediction = self.decoder:forward(torch.Tensor(output))[#output]
    -- prediction contains the probabilities for each word IDs.
    -- The index of the probability is the word ID.
    local prob, wordIds = prediction:topk(5, 1, true, true)

    -- First one is the most likely.
    next_output = wordIds[1]
    
    table.insert(output, next_output)

    -- Terminate on EOS token
    if next_output == self.eosToken then
      break
    end

    table.insert(predictions, wordIds)
    table.insert(probabilities, prob)
  end 

  self.decoder:forget()
  self.encoder:forget()

  return predictions, probabilities
end

----------------------------------------------------------------------------------------------------
-- Beam search
----------------------------------------------------------------------------------------------------
local BEAMS_NUMBER = 2
local MAX_OUTPUT_SIZE_BEAM_SEARCH = 7

function reverse(tbl)
  for i = 1, math.floor(#tbl / 2) do
    local tmp = tbl[i]
    tbl[i] = tbl[#tbl - i + 1]
    tbl[#tbl - i + 1] = tmp
  end
end

function Seq2Seq:beam_search(output, decoder)

  if #output == MAX_OUTPUT_SIZE_BEAM_SEARCH or 
    (#output ~= 0 and output[#output] == self.eosToken) then
    return {}, 0
  end

  local prediction = decoder:forward(torch.Tensor(output))[#output]
  -- prob - word probability from log soft max (negative), the bigger it is the most probable is the corresponding word
  local prob, wordIds = prediction:topk(5, 1, true, true) 
  local beam_prob = {}
  p_sum = 0

  for i = 1, BEAMS_NUMBER do
    table.insert(output, wordIds[i])
    _, beam_prob_cur = Seq2Seq:beam_search(output, decoder)
    beam_prob[i] = beam_prob_cur
    p_sum = p_sum + beam_prob_cur
    table.remove(output)
  end

  p = math.random()
  best_beam_i = BEAMS_NUMBER
  p_sum = 0

  for i = 1, BEAMS_NUMBER do
    beam_prob[i] = beam_prob[i] / p_sum
    p_sum = p_sum + beam_prob[i]

    if p < p_sum then
      best_beam_i = i

      break
    end
  end

  -- local best_beam_i = 1
  -- local best_beam_prob = -1e30


  -- for i = 1, BEAMS_NUMBER do
  --   table.insert(output, wordIds[i])
    -- beamWordIds, beam_prob = Seq2Seq:beam_search(output, decoder)
    -- beam_prob = beam_prob + prob[i]

    -- if beam_prob > best_beam_prob then
    --   best_beam_prob = beam_prob
    --   best_beam_i = i
    -- end
  --   table.remove(output)
  -- end

  table.insert(output, wordIds[best_beam_i])
  beamWordIds, beam_prob = Seq2Seq:beam_search(output, decoder)
  table.remove(output)

  table.insert(beamWordIds, wordIds[best_beam_i])

  return beamWordIds, beam_prob + prob[best_beam_i]
end

function Seq2Seq:eval_with_beam_search(input)
  assert(self.goToken, "No goToken specified")
  assert(self.eosToken, "No eosToken specified")

  self.encoder:forward(input)
  self:forwardConnect(input:size(1))

  -- Forward <go> and all of it's output recursively back to the decoder
  output = {self.goToken}

  wordIds, _ = Seq2Seq:beam_search(output, self.decoder)

  self.decoder:forget()
  self.encoder:forget()

  reverse(wordIds)

  return wordIds
end
