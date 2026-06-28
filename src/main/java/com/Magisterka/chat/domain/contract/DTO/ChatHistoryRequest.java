package com.Magisterka.chat.domain.contract.DTO;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatHistoryRequest {

    private String matchId;
    private String firstTeam;
    private String secondTeam;
}
