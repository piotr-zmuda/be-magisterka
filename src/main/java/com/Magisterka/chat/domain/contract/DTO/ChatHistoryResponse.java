package com.Magisterka.chat.domain.contract.DTO;

import lombok.*;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatHistoryResponse {

    private String matchId;
    private List<ChatMessage> messages;
}
