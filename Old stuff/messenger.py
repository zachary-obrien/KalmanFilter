#!/usr/bin/env python3
#Copyright (c) 2020, Zachary OBrien, All rights reserved.
package apiV2

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	//go get -u github.com/aws/aws-sdk-go

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ses"
	"github.com/gin-gonic/gin"
)

var (
	Sender               = "no-reply@motionloft.com"
	AwsRegion            = "us-west-2"
	CharSet              = "UTF-8"
	ConfigurationSetName = "ConfigSet1"
)

type Notifier struct {
	sess         *session.Session
	accountPhone string
	accountSid   string
	authToken    string
}

func NewNotifier() (*Notifier, error) {
	sess, err := session.NewSession(&aws.Config{
		Region: aws.String(AwsRegion)},
	)

	if err != nil {
		return nil, err
	}

	return &Notifier{
		sess:         sess,
		accountPhone: "19726468633",
		accountSid:   "AC8b62f4aac9505cfe8cbc096c26822ea9",
		authToken:    "15bab803a9c849e4ad0b4a9ce5468ef4",
	}, nil
}

// sendEmail
func (n *Notifier) sendEmail(toField []string, subject, htmlBody, textBody string, trackEmail bool) error {
	// Create an SES client in the session.
	svc := ses.New(n.sess)

	// Assemble the email.
	input := &ses.SendEmailInput{
		Destination: &ses.Destination{
			CcAddresses: []*string{},
			ToAddresses: aws.StringSlice(toField),
		},
		Message: &ses.Message{
			Body: &ses.Body{
				Html: &ses.Content{
					Charset: aws.String(CharSet),
					Data:    aws.String(htmlBody),
				},
				Text: &ses.Content{
					Charset: aws.String(CharSet),
					Data:    aws.String(textBody),
				},
			},
			Subject: &ses.Content{
				Charset: aws.String(CharSet),
				Data:    aws.String(subject),
			},
		},
		Source: aws.String(Sender),
	}
	if trackEmail {
		input.ConfigurationSetName = aws.String(ConfigurationSetName)
	}
	// Attempt to send the email.
	_, err := svc.SendEmail(input)

	// Display error messages if they occur.
	if err != nil {
		if aerr, ok := err.(awserr.Error); ok {
			switch aerr.Code() {
			case ses.ErrCodeMessageRejected:
				fmt.Println(ses.ErrCodeMessageRejected, aerr.Error())
			case ses.ErrCodeMailFromDomainNotVerifiedException:
				fmt.Println(ses.ErrCodeMailFromDomainNotVerifiedException, aerr.Error())
			case ses.ErrCodeConfigurationSetDoesNotExistException:
				fmt.Println(ses.ErrCodeConfigurationSetDoesNotExistException, aerr.Error())
			default:
				fmt.Println(aerr.Error())
			}
			return aerr
		}
		// Print the error, cast err to awserr.Error to get the Code and
		// Message from an error.
		fmt.Println(err.Error())
		return err
	}

	return nil
}

func (n *Notifier) SendEmailHandler(c *gin.Context) {
	toField := c.PostFormArray("to")
	subjectField := c.PostForm("subject")
	bodyField := c.PostForm("body")
	textBodyField := c.PostForm("text_body")
	trackEmailField := c.PostForm("track_email") == "true"

	if len(toField) == 0 || subjectField == "" || bodyField == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "Missing Required Parameters"})
		return
	}
	err := n.sendEmail(toField, subjectField, bodyField, textBodyField, trackEmailField)
	if err != nil {
		c.AbortWithError(http.StatusInternalServerError, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"success": true})
}

// SendSMSHandler
func (n *Notifier) SendSMSHandler(c *gin.Context) {
	bodyField := c.PostForm("body")
	toField := c.PostFormArray("to")

	if len(toField) == 0 || bodyField == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "Missing Required Parameters"})
		return
	}

	err := n.sendSMS(toField, bodyField)
	if err != nil {
		c.AbortWithError(http.StatusInternalServerError, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"success": true})
}

func (n *Notifier) smsUrlStr() string {
	return "https://api.twilio.com/2010-04-01/Accounts/" + n.accountSid + "/Messages.json"
}

// sendSMS
func (n *Notifier) sendSMS(toField []string, bodyField string) error {
	for _, to := range toField {
		msgData := url.Values{}
		msgData.Set("To", to)
		msgData.Set("From", n.accountPhone)
		msgData.Set("Body", bodyField)
		client := &http.Client{}
		req, err := http.NewRequest("POST", n.smsUrlStr(), strings.NewReader(msgData.Encode()))
		if err != nil {
			return err
		}
		req.SetBasicAuth(n.accountSid, n.authToken)
		req.Header.Add("Accept", "application/json")
		req.Header.Add("Content-Type", "application/x-www-form-urlencoded")

		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		var data map[string]interface{}
		decoder := json.NewDecoder(resp.Body)

		if err = decoder.Decode(&data); err != nil {
			return err
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			if msg, ok := data["error_message"]; ok {
				return errors.New(msg.(string))
			}
			if msg, ok := data["message"]; ok {
				return errors.New(msg.(string))
			}
			return fmt.Errorf("Unknown error: status code %v", resp.StatusCode)
		}
	}
	return nil
}


