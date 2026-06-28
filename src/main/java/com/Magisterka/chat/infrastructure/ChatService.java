package com.Magisterka.chat.infrastructure;

import com.Magisterka.chat.domain.adapter.ChatAdapter;
import com.Magisterka.chat.domain.contract.DTO.ChatAnalyseRequest;
import com.Magisterka.chat.domain.contract.DTO.ChatAnalyseResponse;
import com.Magisterka.chat.domain.contract.DTO.ChatHistoryRequest;
import com.Magisterka.chat.domain.contract.DTO.ChatHistoryResponse;
import com.Magisterka.chat.infrastructure.repository.ChatRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class ChatService implements ChatAdapter {

    private final ChatRepository chatRepository;

    @Override
    public ChatAnalyseResponse sendToAnalyse(ChatAnalyseRequest request) {
        if (request == null || request.getMessage() == null || request.getMessage().isBlank()) {
            throw new IllegalArgumentException("Message is required");
        }

        return chatRepository.sendToAnalyse(request);
    }

    @Override
    public ChatHistoryResponse getHistory(ChatHistoryRequest request) {
        if (request == null || request.getMatchId() == null || request.getMatchId().isBlank()) {
            throw new IllegalArgumentException("Match id is required");
        }

        return chatRepository.getHistory(request);
    }
}
