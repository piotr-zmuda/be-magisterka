package com.Magisterka.shared.config;


import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

@Configuration
public class ClientConfig {

    @Bean
    RestClient serviceARestClient(@Value("${service.main.url}") String baseUrl) {
        String root = baseUrl.trim().replaceAll("/+$", "");

        CloseableHttpClient httpClient = HttpClients.custom()
                .disableRedirectHandling()
                .build();

        HttpComponentsClientHttpRequestFactory requestFactory =
                new HttpComponentsClientHttpRequestFactory(httpClient);

        return RestClient.builder()
                .requestFactory(requestFactory)
                .baseUrl(root)
                .build();
    }


}
