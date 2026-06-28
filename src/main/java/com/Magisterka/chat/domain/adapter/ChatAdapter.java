package com.Magisterka.chat.domain.adapter;

import com.Magisterka.chat.domain.contract.DTO.ChatAnalyseRequest;
import com.Magisterka.chat.domain.contract.DTO.ChatAnalyseResponse;
import com.Magisterka.chat.domain.contract.DTO.ChatHistoryRequest;
import com.Magisterka.chat.domain.contract.DTO.ChatHistoryResponse;

public interface ChatAdapter {

    ChatAnalyseResponse sendToAnalyse(ChatAnalyseRequest request);

    ChatHistoryResponse getHistory(ChatHistoryRequest request);
}
