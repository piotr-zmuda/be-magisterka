package com.Magisterka.shared;


import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.client.RestClientResponseException;

@RestControllerAdvice
public class GlobalDownstreamErrorHandler {

    @ExceptionHandler(RestClientResponseException.class)
    public ResponseEntity<String> handleRestClientResponse(RestClientResponseException e) {
        return ResponseEntity
                .status(e.getStatusCode())
                .contentType(MediaType.APPLICATION_JSON)
                .body(e.getResponseBodyAsString());
    }
}
