package com.Magisterka.chat.controller;

import com.Magisterka.chat.domain.contract.ChatResources;
import com.Magisterka.chat.domain.contract.DTO.ChatAnalyseRequest;
import com.Magisterka.chat.domain.contract.DTO.ChatAnalyseResponse;
import com.Magisterka.chat.domain.contract.DTO.ChatHistoryRequest;
import com.Magisterka.chat.domain.contract.DTO.ChatHistoryResponse;
import com.Magisterka.chat.infrastructure.ChatService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class ChatController implements ChatResources {

    private final ChatService chatService;

    @Override
    public ChatAnalyseResponse sendToAnalyse(ChatAnalyseRequest request) {
        return chatService.sendToAnalyse(request);
    }

    @Override
    public ChatHistoryResponse getHistory(ChatHistoryRequest request) {
        return chatService.getHistory(request);
    }
}
