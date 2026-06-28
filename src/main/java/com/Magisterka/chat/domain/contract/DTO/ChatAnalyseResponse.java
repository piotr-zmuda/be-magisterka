package com.Magisterka.chat.domain.contract.DTO;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatAnalyseResponse {

    private String answer;
    private String model;
}
