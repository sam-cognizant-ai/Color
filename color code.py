import json

def substring_search(input_json):
    main_string = input_json.get('main', '')
    sub_string = input_json.get('sub', '')

    if not isinstance(main_string, str) or not isinstance(sub_string, str):
        return {"error": "Both 'main' and 'sub' should be strings."}

    start_index = main_string.find(sub_string)
    if start_index == -1:
        return {"error": f"'{sub_string}' not found in 'main'."}

    end_index = start_index + len(sub_string) - 1
    return {"start_index": start_index, "end_index": end_index}

# Example Usage
input_data = {
    "main": "CSR: Thank you for calling ABC Travel, this is John speaking. How may I assist you today?, Customer: Yes, I need to book a flight to Paris next week. I tried doing it online but your website is completely useless. Can you help me?, CSR: Absolutely, I'd be happy to assist with booking your flight to Paris. May I first get your name and contact information?, Customer: It's Mark Johnson. My number is 555-1234. Why do you need all that? Just book my flight., CSR: Thank you Mr. Johnson. I need some basic information to pull up your reservation. What dates were you looking to travel?, Customer: I told you, next week. Don't you listen? I need to be in Paris on the 15th so book it for the 14th. , CSR: Okay, give me one moment while I search for flight options on the 14th. Will you be traveling roundtrip? And could you confirm your departure airport?, Customer: Yes, roundtrip from LAX. This is taking forever, you should have had this done already., CSR: My apologies for the delay Mr. Johnson, I'm pulling up options now. I see a nonstop flight on Air France departing LAX at 9:35am and arriving Paris at 5:50pm on the 14th. The return flight departs Paris at 12:15pm on the 21st. Would that work for you?, Customer: I guess so. But I wanted first class. Do you have anything in first class?, CSR: Let me check on first class availability for you. On the outbound, there is a first class seat available on the 7pm flight arriving at 1:15pm the next day. On the return, there is first class availability on the flight departing Paris at 3:45pm arriving LAX at 6:55pm. Would you like me to book those first class seats for you?, Customer: Finally, yes. But I better get a good seat, not one of those cramped ones. And I need a meal on both flights. , CSR: No problem, I can request premium first class seating for you with extra legroom. And first class tickets include meal service. Let me just confirm - I have you booked first class roundtrip from LAX to Paris. Departing LAX on the 14th at 7pm, returning from Paris on the 21st at 3:45pm. Is that correct?, Customer: I think so. Just hurry up and book it before I change my mind., CSR: Booking your flights now. Okay, your first class tickets are booked. I'll email you the itinerary along with your e-ticket numbers. The total fare is $9,550 which I can charge to the card on file. Do you have any other questions?, Customer: No, that's it. This took way too long but hopefully you got it right. I'll be calling back if there are any issues. CSR: We appreciate you choosing ABC Travel. Thank you for your patience Mr. Johnson, and have a wonderful trip!",
    "sub": "I tried doing it online but your website is completely useless."
}

result = substring_search(input_data)
print(json.dumps(result, indent=4))
