package com.Magisterka.chat.domain.contract.DTO;

import lombok.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class OllamaGenerateRequest {

    private String model;
    private String prompt;
    private Boolean stream;
}
