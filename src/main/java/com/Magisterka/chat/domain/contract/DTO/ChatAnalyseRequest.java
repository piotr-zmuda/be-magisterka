package com.Magisterka.chat.domain.contract.DTO;

import lombok.*;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatAnalyseRequest {

    private String message;
    private String matchId;
    private List<ChatMessage> history;
}
