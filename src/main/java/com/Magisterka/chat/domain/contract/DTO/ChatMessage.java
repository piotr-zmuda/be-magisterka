package com.Magisterka.chat.domain.contract.DTO;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatMessage {

    private String role;
    private String content;
}
