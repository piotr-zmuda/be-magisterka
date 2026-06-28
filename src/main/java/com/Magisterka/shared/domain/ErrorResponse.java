package com.Magisterka.shared.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@NoArgsConstructor
@AllArgsConstructor
@Data
@Builder
public class ErrorResponse implements Serializable {

    private int status;
    private String error;
    private String message;

    public static ErrorResponse from(String msg, int status, String error) {
        return ErrorResponse.builder()
                .status(status)
                .error(error)
                .message(msg)
                .build();
    }
}
